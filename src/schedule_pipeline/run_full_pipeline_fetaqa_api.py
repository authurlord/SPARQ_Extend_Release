#!/usr/bin/env python3
"""
H-STAR Pipeline for WikiTQ - vLLM API Version
Uses async API calls instead of loading vLLM locally
"""

import os
import sys
import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import torch
import faiss
from rank_bm25 import BM25Okapi

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.schedule_utils import (
    load_data_split, table_to_str, table_to_str_sql,
    find_intersection_and_add_row_id, Prepare_Data_for_Operator_Sequence,
    format_document, batch_rerank_scores, ROLLBACK,
    merge_clean_and_format_df_dict, retrieve_rows_by_subtables,
    process_error_analysis_list
)
from utils.evaluator import Evaluator
from utils.prompt_generate import (
    build_wikitq_prompt_from_df, evaluate_predictions,
    filter_dataframe_from_responses, fix_sql_query,
    match_subtables, retrieve_rows_by_subtables
)
from utils.multi_db_v2 import NeuralDB, Executor
from utils.normalizer import convert_df_type
from FlagEmbedding import FlagReranker, FlagModel

# Import async LLM client
from utils.async_llm import infer_prompts


# ============================================================================
# 1. Preprocessing Functions (convert_df_type_parallel.py logic)
# ============================================================================

# Global variable for worker processes
worker_dataset = None

def init_worker(dataset_to_share):
    """Initialize worker with dataset"""
    global worker_dataset
    worker_dataset = dataset_to_share

def process_table_at_index(index: int) -> Tuple[int, pd.DataFrame]:
    """Process a single table from the worker's global dataset"""
    original_df = pd.DataFrame(
        worker_dataset[index]['table']['rows'],
        columns=worker_dataset[index]['table']['header']
    )
    processed_df = convert_df_type(original_df)
    return (index, processed_df)

def preprocess_tables_parallel(dataset, num_workers: int = None) -> Dict[int, pd.DataFrame]:
    """
    Parallel preprocessing of tables - DIRECT FUNCTION instead of subprocess
    Returns: Dictionary mapping index -> processed DataFrame
    """
    print(f"  Preprocessing {len(dataset)} tables in parallel...")
    
    if num_workers is None:
        num_workers = cpu_count()
    
    processed_tables_dict = {}
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(dataset,)) as pool:
        results_iterator = pool.imap_unordered(process_table_at_index, range(len(dataset)))
        
        for index, processed_df in tqdm(results_iterator, total=len(dataset), desc="  Processing tables"):
            processed_tables_dict[index] = processed_df
    
    print(f"  Preprocessing complete: {len(processed_tables_dict)} tables")
    return processed_tables_dict


# ============================================================================
# 2. Router Inference Functions (inference_router.py logic)
# ============================================================================

def dataframe_to_llm_string(df: pd.DataFrame) -> str:
    """Convert pandas DataFrame to LLM-readable markdown format"""
    header = "col : " + " | ".join(map(str, df.columns))
    rows = []
    for index, row in df.iterrows():
        row_values = [str(x) for x in row.values]
        row_str = "row " + str(index) + " : " + " | ".join(row_values)
        rows.append(row_str)
    return header + "\n" + "\n".join(rows)

def router_inference_direct(semantic_router: Dict, router_model) -> List[Dict]:
    """
    Direct router inference - DIRECT FUNCTION instead of subprocess
    
    Args:
        semantic_router: Dictionary with keys {index: {'query', 'title', 'table', 'label'}}
        router_model: Pre-loaded FlagModel instance
    
    Returns:
        List of dictionaries with inference results
    """
    print("  Running router inference...")
    model = router_model
    
    # Define all possible labels
    ALL_LABELS = [
        'Base', 'Select_Row', 'Select_Column', 'Execute_SQL', 'RAG_20_5'
    ]
    
    # Pre-encode all labels
    print("  Encoding labels...")
    label_embeddings_np = model.encode(ALL_LABELS)
    label_embeddings = torch.from_numpy(label_embeddings_np)
    print("  Labels encoded")
    
    # Convert dict to list
    inference_data = list(semantic_router.values())
    
    # Prepare all queries
    queries_to_encode = []
    for item in inference_data:
        original_query = item['query']
        title = item['title']
        table_df = item['table']
        table_str = dataframe_to_llm_string(table_df)
        base_query_format = "Query: {} [SEP] Table Title: {} [SEP] Table: {}"
        model_query = base_query_format.format(original_query, title, table_str)
        queries_to_encode.append(model_query)
    
    # Batch encode all queries
    print("  Encoding queries...")
    query_embeddings_np = model.encode(queries_to_encode, batch_size=4, max_length=2048)
    query_embeddings = torch.from_numpy(query_embeddings_np)
    print("  Queries encoded")
    
    # Calculate similarity
    print("  Computing similarities...")
    similarities_matrix = query_embeddings @ label_embeddings.T
    
    # Integrate results
    results_data = []
    for i, item in enumerate(tqdm(inference_data, desc="  Integrating results")):
        similarities = similarities_matrix[i].cpu().tolist()
        label_scores = {label: score for label, score in zip(ALL_LABELS, similarities)}
        
        new_item = item.copy()
        new_item['result'] = label_scores
        results_data.append(new_item)
    
    print("  Router inference complete")
    return results_data


# ============================================================================
# 3. Hybrid Retrieval Functions (Hybrid_Retrieve.py logic)
# ============================================================================

class HybridTableRetriever:
    """Hybrid (sparse + dense) retrieval system"""
    
    def __init__(self, embedding_model, alpha: float = 0.5, view_samples: int = 10):
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1.")
        self.model = embedding_model
        self.alpha = alpha
        self.view_samples = view_samples
    
    def _generate_column_views(self, df: pd.DataFrame, table_title: str) -> dict:
        """Generate text views for each column"""
        column_views = defaultdict(list)
        for col_name in df.columns:
            col = df[col_name].dropna()
            if col.empty: 
                continue
            column_views[col_name].append(f"Column named '{col_name}' in the table '{table_title}'.")
            if pd.api.types.is_numeric_dtype(col):
                stats = f"Numerical column with values from {col.min()} to {col.max()}."
                column_views[col_name].append(f"'{col_name}': {stats}")
                samples = col.sample(min(len(col), self.view_samples)).tolist()
                column_views[col_name].append(f"'{col_name}' has sample values like {samples}.")
            else:
                top_values = col.value_counts().nlargest(self.view_samples).index.tolist()
                column_views[col_name].append(f"'{col_name}' has common values like: {top_values}.")
                samples = col.sample(min(len(col), self.view_samples)).tolist()
                column_views[col_name].append(f"'{col_name}' contains examples like: {samples}.")
        return dict(column_views)
    
    def _generate_row_views(self, df: pd.DataFrame, table_title: str, columns_to_use: list) -> list:
        """Generate text view for each row"""
        row_views = []
        for index, row in df.iterrows():
            row_parts = [f"{col} is {row[col]}" for col in columns_to_use if col in row]
            row_values_str = "; ".join(row_parts)
            final_text = f"From table '{table_title}', row {index}: {row_values_str}."
            row_views.append(final_text)
        return row_views
    
    def _normalize_scores(self, scores: dict) -> dict:
        """Normalize scores to [0, 1] range"""
        if not scores or len(scores) == 1:
            return {k: 1.0 for k in scores}
        
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
    
    def _fuse_scores(self, dense_scores: dict, sparse_scores: dict) -> dict:
        """Fuse normalized dense and sparse scores"""
        dense_norm = self._normalize_scores(dense_scores)
        sparse_norm = self._normalize_scores(sparse_scores)
        
        fused = defaultdict(float)
        all_keys = set(dense_norm.keys()) | set(sparse_norm.keys())
        
        for key in all_keys:
            dense_score = dense_norm.get(key, 0)
            sparse_score = sparse_norm.get(key, 0)
            fused[key] = (self.alpha * dense_score) + ((1 - self.alpha) * sparse_score)
        
        return dict(fused)
    
    def retrieve(self, 
                 rewrite_queries: list, 
                 tables: list, 
                 table_titles: list, 
                 max_rows_m: int, 
                 max_cols_n: int) -> list:
        """Main retrieval method"""
        if not (len(rewrite_queries) == len(tables) == len(table_titles)):
            raise ValueError("Input lists must have the same length.")
        
        num_tasks = len(rewrite_queries)
        
        # Column selection
        selected_columns_per_table = []
        for i in range(num_tasks):
            table = tables[i]
            if table.shape[1] <= max_cols_n:
                selected_columns_per_table.append(table.columns.tolist())
                continue
            
            col_views_dict = self._generate_column_views(table, table_titles[i])
            col_views_flat = [view for views in col_views_dict.values() for view in views]
            flat_idx_to_col_name = [name for name, views in col_views_dict.items() for _ in views]
            
            # Sparse retrieval (BM25)
            tokenized_corpus = [doc.lower().split() for doc in col_views_flat]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = rewrite_queries[i].lower().split()
            sparse_scores_flat = bm25.get_scores(tokenized_query)
            
            sparse_col_scores = defaultdict(list)
            for idx, score in enumerate(sparse_scores_flat):
                sparse_col_scores[flat_idx_to_col_name[idx]].append(score)
            sparse_col_scores_avg = {k: np.mean(v) for k, v in sparse_col_scores.items()}
            
            # Dense retrieval (FAISS)
            col_view_embeddings = self.model.encode(col_views_flat, batch_size=256)
            query_embedding = self.model.encode([rewrite_queries[i]])
            
            col_view_embeddings = col_view_embeddings.astype(np.float32)
            query_embedding = query_embedding.astype(np.float32)
            
            dim = query_embedding.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(col_view_embeddings)
            index.add(col_view_embeddings)
            faiss.normalize_L2(query_embedding)
            
            scores, indices = index.search(query_embedding, k=len(col_views_flat))
            
            dense_col_scores = defaultdict(list)
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    dense_col_scores[flat_idx_to_col_name[idx]].append(score)
            dense_col_scores_avg = {k: np.mean(v) for k, v in dense_col_scores.items()}
            
            # Fuse scores
            fused_scores = self._fuse_scores(dense_col_scores_avg, sparse_col_scores_avg)
            sorted_cols_by_relevance = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            top_n_cols_by_relevance = {col for col, score in sorted_cols_by_relevance[:max_cols_n]}
            
            original_cols = table.columns.tolist()
            ordered_selected_cols = [col for col in original_cols if col in top_n_cols_by_relevance]
            selected_columns_per_table.append(ordered_selected_cols)
        
        # Row selection
        final_subtables = []
        for i in range(num_tasks):
            table = tables[i]
            selected_cols = selected_columns_per_table[i]
            
            if table.shape[0] <= max_rows_m:
                final_subtables.append(table[selected_cols])
                continue
            
            row_views = self._generate_row_views(table, table_titles[i], selected_cols)
            
            # Sparse retrieval (BM25)
            tokenized_corpus = [doc.lower().split() for doc in row_views]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = rewrite_queries[i].lower().split()
            sparse_row_scores = {idx: score for idx, score in enumerate(bm25.get_scores(tokenized_query))}
            
            # Dense retrieval (FAISS)
            row_embeddings = self.model.encode(row_views, batch_size=256)
            query_embedding = self.model.encode([rewrite_queries[i]])
            
            row_embeddings = row_embeddings.astype(np.float32)
            query_embedding = query_embedding.astype(np.float32)
            
            dim = query_embedding.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(row_embeddings)
            index.add(row_embeddings)
            faiss.normalize_L2(query_embedding)
            scores, indices = index.search(query_embedding, k=table.shape[0])
            dense_row_scores = {idx: score for score, idx in zip(scores[0], indices[0]) if idx != -1}
            
            # Fuse scores
            fused_scores = self._fuse_scores(dense_row_scores, sparse_row_scores)
            sorted_rows_by_relevance = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            
            top_m_view_indices = [row_idx for row_idx, score in sorted_rows_by_relevance[:max_rows_m]]
            
            original_indices = table.index.tolist()
            top_m_original_indices_by_relevance = [original_indices[view_idx] for view_idx in top_m_view_indices]
            
            top_m_original_indices_by_relevance.sort()
            ordered_selected_indices = top_m_original_indices_by_relevance
            
            final_subtables.append(table.loc[ordered_selected_indices, selected_cols])
        
        return final_subtables

def hybrid_retrieve_direct(indices_to_process: List[int],
                          dataset,
                          wikitq_df_processed: Dict[int, pd.DataFrame],
                          embedding_model,
                          max_rows: int = 20,
                          max_cols: int = 5) -> Dict[int, pd.DataFrame]:
    """
    Direct hybrid retrieval - DIRECT FUNCTION instead of subprocess
    
    Args:
        indices_to_process: List of dataset indices to process
        dataset: The dataset
        wikitq_df_processed: Dictionary of preprocessed tables
        embedding_model: Pre-loaded FlagModel instance
        max_rows: Maximum rows to retrieve
        max_cols: Maximum columns to retrieve
    
    Returns:
        Dictionary mapping index -> retrieved DataFrame
    """
    print(f"  Running hybrid retrieval on {len(indices_to_process)} samples...")
    retriever = HybridTableRetriever(embedding_model=embedding_model, alpha=0.5)
    
    # Prepare batch data
    print(f"  Preparing batch data for {len(indices_to_process)} items...")
    batch_queries = [dataset[i]['question'] for i in indices_to_process]
    batch_tables = [wikitq_df_processed[i] for i in indices_to_process]
    batch_titles = [dataset[i]['table']['page_title'] for i in indices_to_process]
    
    # Run retrieval
    print(f"  Running hybrid retrieval...")
    start_time = time.time()
    subtables = retriever.retrieve(
        rewrite_queries=batch_queries,
        tables=batch_tables,
        table_titles=batch_titles,
        max_rows_m=max_rows,
        max_cols_n=max_cols
    )
    elapsed = time.time() - start_time
    print(f"  Retrieval complete in {elapsed:.2f}s")
    
    # Map results back to original indices
    retrieved_tables = {}
    for i, sub_df in enumerate(subtables):
        original_index = indices_to_process[i]
        retrieved_tables[original_index] = sub_df
    
    return retrieved_tables


# ============================================================================
# 4. API-based LLM Functions
# ============================================================================

def response_vllm_api(all_instructions: List[str], sample_num: int,
                     api_base: str, api_key: str, model_name: str,
                     temperature: float = 0.7, top_p: float = 0.8,
                     concurrency: int = 128, max_tokens: int = 2048,
                     cache_file: str = None) -> List[List[str]]:
    """
    Generate responses using vLLM API (async client)

    Args:
        all_instructions: List of prompts
        sample_num: Number of samples per prompt (n parameter)
        api_base: API base URL
        api_key: API key
        model_name: Model name on the server
        temperature: Sampling temperature
        top_p: Sampling top_p
        concurrency: Max concurrent requests
        max_tokens: Max tokens per response
        cache_file: Optional path to persist/resume this stage's responses.
                    Keyed by an md5 of the exact prompts + sample_num; any
                    mismatch or load error falls back to regenerating, so it
                    is correctness-neutral (only skips re-calling the API when
                    the identical prompts were already answered).

    Returns:
        List of lists of generated texts
    """
    import json as _json, os as _os, hashlib as _hashlib
    _key = _hashlib.md5(("\x00".join(all_instructions)).encode("utf-8", "ignore")).hexdigest()
    if cache_file and _os.path.exists(cache_file):
        try:
            with open(cache_file) as _f:
                _cached = _json.load(_f)
            if (_cached.get("key") == _key
                    and _cached.get("sample_num") == sample_num
                    and len(_cached.get("results", [])) == len(all_instructions)):
                print(f"  [cache] Resumed {len(all_instructions)} responses from {cache_file}")
                return _cached["results"]
            print(f"  [cache] Stale cache at {cache_file}; regenerating")
        except Exception as _e:
            print(f"  [cache] Could not load {cache_file} ({type(_e).__name__}); regenerating")

    print(f"  Calling vLLM API with {len(all_instructions)} prompts (n={sample_num})...")

    # Call async API
    results, metrics_rows, summary = infer_prompts(
        prompts=all_instructions,
        llm_path=model_name,
        llm_name=model_name,
        api_base=api_base,
        api_key=api_key,
        concurrency=concurrency,
        request_timeout=300,
        max_retries=5,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        sample_num=sample_num,
        presence_penalty=0.0,
        show_progress=True
    )
    
    print(f"  API call complete:")
    print(f"    - Duration: {summary['batch_dur']:.2f}s")
    print(f"    - QPS: {summary['qps']:.2f}")
    print(f"    - Input tokens/s: {summary['in_tps']:.2f}")
    print(f"    - Output tokens/s: {summary['out_tps']:.2f}")

    if cache_file:
        try:
            _os.makedirs(_os.path.dirname(cache_file) or ".", exist_ok=True)
            _tmp = cache_file + ".tmp"
            with open(_tmp, "w") as _f:
                _json.dump({"key": _key, "sample_num": sample_num, "results": results}, _f)
            _os.replace(_tmp, cache_file)
            print(f"  [cache] Saved {len(results)} responses to {cache_file}")
        except Exception as _e:
            print(f"  [cache] Could not save {cache_file} ({type(_e).__name__})")

    return results


def load_models_no_vllm(args):
    """
    Pre-load models (without vLLM, API version)
    
    Returns:
        Dict containing models:
        - 'router_model': FlagModel for router
        - 'embedding_model': FlagModel for retrieval
        - 'reranker_model': FlagReranker for check model
    """
    print("\n[Model Loading] Pre-loading embedding models (no local vLLM)...")
    models = {}
    
    # 1. Load router model (FlagModel)
    print("\n  [1/3] Loading router model...")
    instruction = "What kind of operation should I take to better filter relevant tables to complete the QA? "
    try:
        models['router_model'] = FlagModel(
            args.router_model_path,
            query_instruction_for_retrieval=instruction,
            use_fp16=True
        )
        print("    ✓ Router model loaded on GPU")
    except Exception as e:
        print(f"    ⚠ GPU loading failed ({type(e).__name__}), trying CPU...")
        models['router_model'] = FlagModel(
            args.router_model_path,
            query_instruction_for_retrieval=instruction,
            use_fp16=False
        )
        print("    ✓ Router model loaded on CPU")
    
    # 2. Load embedding model (FlagModel)
    print("\n  [2/3] Loading embedding model...")
    try:
        models['embedding_model'] = FlagModel(
            args.embedding_model_path,
            use_fp16=True
        )
        print("    ✓ Embedding model loaded on GPU")
    except Exception as e:
        print(f"    ⚠ GPU loading failed ({type(e).__name__}), trying CPU...")
        models['embedding_model'] = FlagModel(
            args.embedding_model_path,
            use_fp16=False
        )
        print("    ✓ Embedding model loaded on CPU")
    
    # 3. Load reranker model (FlagReranker) - Always use CPU
    print("\n  [3/3] Loading reranker model (on CPU)...")
    try:
        models['reranker_model'] = FlagReranker(
            args.check_model_path,
            use_fp16=False,
            device='cpu'
        )
    except TypeError:
        models['reranker_model'] = FlagReranker(
            args.check_model_path,
            use_fp16=False
        )
    print("    ✓ Reranker model loaded on CPU")
    
    print("\n" + "="*80)
    print("✓ All models loaded successfully!")
    print("="*80 + "\n")
    return models


# ============================================================================
# 5. Main Pipeline
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="H-STAR Full Pipeline for WikiTQ (API Version)")
    
    # API parameters
    parser.add_argument('--api_base', type=str, 
                       default='http://127.0.0.1:8000/v1',
                       help='vLLM API base URL')
    parser.add_argument('--api_key', type=str, 
                       default='EMPTY',
                       help='API key (vLLM server may ignore)')
    parser.add_argument('--model_name', type=str,
                       default='Qwen2.5-7B-Instruct',
                       help='Model name on the API server')
    parser.add_argument('--concurrency', type=int, default=128,
                       help='Max concurrent API requests')
    
    # Model paths (embedding models only)
    parser.add_argument('--embedding_model_path', type=str,
                       default='/data/workspace/yanmy/models/bge-m3',
                       help='Path to embedding model')
    parser.add_argument('--router_model_path', type=str,
                       default='/data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned/',
                       help='Path to router model')
    parser.add_argument('--check_model_path', type=str,
                       default='/data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned/',
                       help='Path to check model')
    
    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default='fetaqa',
                       choices=['fetaqa'],
                       help='Dataset name')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/wikitq_api',
                       help='Temporary save path for intermediate results')
    
    # Pipeline parameters
    parser.add_argument('--tau', type=float, default=0.82,
                       help='Router threshold')
    parser.add_argument('--check_tau', type=float, default=0.8,
                       help='Check model threshold')
    parser.add_argument('--n_parallel', type=int, default=32,
                       help='Number of parallel workers for preprocessing')
    
    # Sampling parameters
    parser.add_argument('--select_sample_num', type=int, default=2,
                       help='Number of samples for Select_Row/Select_Column')
    parser.add_argument('--sql_sample_num', type=int, default=3,
                       help='Number of samples for Execute_SQL')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8,
                       help='Sampling top_p')
    parser.add_argument('--max_tokens', type=int, default=2048,
                       help='Max tokens per generation')
    
    # Execution control
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing if already done')
    parser.add_argument('--skip_router', action='store_true',
                       help='Skip router inference if already done')
    parser.add_argument('--skip_rag', action='store_true',
                       help='Skip RAG if already done')
    parser.add_argument('--first_n', type=int, default=-1,
                       help='Only process first N samples (-1 for all)')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate results to disk (for debugging)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("H-STAR Full Pipeline for WikiTQ (API Version)")
    print("="*80)
    print(f"API Base: {args.api_base}")
    print(f"Model: {args.model_name}")
    print(f"Max Concurrency: {args.concurrency}")
    print("="*80)
    
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    ALL_LABELS = ['Base', 'Select_Row', 'Select_Column', 'Execute_SQL', 'RAG_20_5']
    
    # Timeline tracking
    timeline: Dict[str, float] = {}
    pipeline_start_all = time.perf_counter()
    
    # Load models (no vLLM)
    _t0 = time.perf_counter()
    models = load_models_no_vllm(args)
    timeline['Model Loading'] = time.perf_counter() - _t0
    print(f"  [Timing] Model Loading: {timeline['Model Loading']:.2f}s")
    
    router_model = models['router_model']
    embedding_model = models['embedding_model']
    reranker_model = models['reranker_model']

    start_time = time.perf_counter()
    
    # ========================================================================
    # Step 1: Data Preprocessing
    # ========================================================================
    print("\n[Step 1] Data Preprocessing...")
    
    # Load raw dataset
    _t = time.perf_counter()
    dataset = load_data_split(args.dataset_name, args.split)
    
    if args.first_n > 0:
        dataset = dataset.select(range(min(args.first_n, len(dataset))))
        print(f"Processing only first {args.first_n} samples")
    
    print(f"Loaded {len(dataset)} samples")
    timeline['Data Load'] = time.perf_counter() - _t
    print(f"  [Timing] Data Load: {timeline['Data Load']:.2f}s")
    
    _t1 = time.perf_counter()
    # Check if we should skip preprocessing
    preprocess_cache = f'{args.tmp_save_path}/wikitq_df_processed.npy'
    cache_exists = os.path.exists(preprocess_cache)
    
    if cache_exists:
        print(f"  ✓ Found cached preprocessed tables: {preprocess_cache}")
        if not args.skip_preprocess:
            print("    (Use --skip_preprocess to load from cache, or delete cache to reprocess)")
        print("  Loading from cache...")
        wikitq_df_processed = np.load(preprocess_cache, allow_pickle=True).item()
        print(f"  Loaded {len(wikitq_df_processed)} preprocessed tables from cache")
    else:
        if args.skip_preprocess:
            print(f"  ⚠ Warning: --skip_preprocess set but cache not found at {preprocess_cache}")
            print("  Will proceed with preprocessing...")
        
        # Direct function call instead of subprocess
        print("  Preprocessing tables...")
        wikitq_df_processed = preprocess_tables_parallel(dataset, args.n_parallel)
        
        # Optionally save (or always save for future use)
        if args.save_intermediate or not cache_exists:
            print(f"  Saving preprocessed tables to {preprocess_cache}...")
            np.save(preprocess_cache, wikitq_df_processed)
            print(f"  ✓ Saved preprocessed tables (use --skip_preprocess next time to load from cache)")
    timeline['Step 1 - Preprocessing'] = time.perf_counter() - _t1
    print(f"  [Timing] Step 1 - Preprocessing: {timeline['Step 1 - Preprocessing']:.2f}s")
    
    # ========================================================================
    # Step 2: Build Router Query (in memory)
    # ========================================================================
    print("\n[Step 2] Building Router Query...")
    _t2 = time.perf_counter()
    semantic_router = {}
    for index in tqdm(range(len(dataset)), desc="  Building queries"):
        semantic_router[index] = {
            'query': dataset[index]['question'],
            'title': dataset[index]['table']['page_title'],
            'table': wikitq_df_processed[index],
            'label': []
        }
    
    if args.save_intermediate:
        with open(f'{args.tmp_save_path}/router_query.pkl', 'wb') as f:
            pickle.dump(semantic_router, f)
    timeline['Step 2 - Build Router Query'] = time.perf_counter() - _t2
    print(f"  [Timing] Step 2 - Build Router Query: {timeline['Step 2 - Build Router Query']:.2f}s")
    
    # ========================================================================
    # Step 3: Construct Database (in memory)
    # ========================================================================
    print("\n[Step 3] Constructing Database...")
    _t3 = time.perf_counter()
    table_titles = [dataset[i]['table']['page_title'] for i in range(len(dataset))]
    db = NeuralDB(tables=wikitq_df_processed, table_titles=table_titles)
    executor = Executor()
    print("  Database initialized")
    timeline['Step 3 - Construct Database'] = time.perf_counter() - _t3
    print(f"  [Timing] Step 3 - Construct Database: {timeline['Step 3 - Construct Database']:.2f}s")
    
    # ========================================================================
    # Step 4: Router Model Inference
    # ========================================================================
    print("\n[Step 4] Router Model Inference...")
    _t4 = time.perf_counter()
    
    router_cache = f'{args.tmp_save_path}/inference_result.pkl'
    if args.skip_router and os.path.exists(router_cache):
        print("  Loading cached router results...")
        with open(router_cache, 'rb') as f:
            error_analysis_row = pickle.load(f)
    else:
        # Direct function call with pre-loaded model
        error_analysis_row = router_inference_direct(semantic_router, router_model)
        
        # Optionally save
        if args.save_intermediate:
            with open(router_cache, 'wb') as f:
                pickle.dump(error_analysis_row, f)
            print(f"  Saved router results to {router_cache}")
    timeline['Step 4 - Router Inference'] = time.perf_counter() - _t4
    print(f"  [Timing] Step 4 - Router Inference: {timeline['Step 4 - Router Inference']:.2f}s")
    
    # ========================================================================
    # Step 5: Parse Router Results & Organize LLM Query List
    # ========================================================================
    print("\n[Step 5] Parsing Router Results...")
    _t5 = time.perf_counter()
    ranked_result = process_error_analysis_list(
        error_analysis_row, truncate=True, tau=args.tau
    )
    
    print("  Router result distribution:")
    print(pd.DataFrame([str(r) for r in ranked_result.values()]).value_counts())
    
    # Organize LLM query list
    LLM_query_list = {}
    for method in ALL_LABELS:
        LLM_query_list[method] = {
            'index': [],
            'query': [],
            'qa': [],
            'response': []
        }
    
    for index in tqdm(range(len(dataset)), desc="  Organizing queries"):
        for method in ALL_LABELS:
            if method in ranked_result[index]:
                LLM_query_list[method]['index'].append(index)
                
                if method == 'Select_Column':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[index], index,
                        template_path='../prompts/col_select_sql.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].append(prompt)
                
                elif method == 'Select_Row':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[index], index,
                        template_path='../prompts/row_select_sql.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].append(prompt)
                
                elif method == 'Execute_SQL':
                    prompt = build_wikitq_prompt_from_df(
                        dataset, wikitq_df_processed[index], index,
                        template_path='../prompts/sql_reason_fetaqa.txt',
                        processed=True
                    )
                    LLM_query_list[method]['query'].append(prompt)
    
    print(f"  Query counts:")
    for method in ALL_LABELS:
        print(f"    {method}: {len(LLM_query_list[method]['index'])}")
    timeline['Step 5 - Parse Router & Build Queries'] = time.perf_counter() - _t5
    print(f"  [Timing] Step 5 - Parse Router & Build Queries: {timeline['Step 5 - Parse Router & Build Queries']:.2f}s")
    
    # ========================================================================
    # Step 6: Execute RAG Task
    # ========================================================================
    rag_indices = LLM_query_list['RAG_20_5']['index']
    
    if len(rag_indices) > 0:
        print(f"\n[Step 6] Executing RAG on {len(rag_indices)} samples...")
        _t6 = time.perf_counter()
        
        rag_cache = f'{args.tmp_save_path}/Hybrid_Retrieve_output.npy'
        if args.skip_rag and os.path.exists(rag_cache):
            print("  Loading cached RAG results...")
            RAG_20_5 = np.load(rag_cache, allow_pickle=True).item()
        else:
            # Direct function call with pre-loaded model
            RAG_20_5 = hybrid_retrieve_direct(
                indices_to_process=rag_indices,
                dataset=dataset,
                wikitq_df_processed=wikitq_df_processed,
                embedding_model=embedding_model,
                max_rows=20,
                max_cols=5
            )
            
            # Optionally save
            if args.save_intermediate:
                np.save(rag_cache, RAG_20_5)
                print(f"  Saved RAG results to {rag_cache}")
    else:
        print("\n[Step 6] No RAG samples, skipping...")
        RAG_20_5 = {}
        _t6 = time.perf_counter()
    
    timeline['Step 6 - RAG'] = time.perf_counter() - _t6
    print(f"  [Timing] Step 6 - RAG: {timeline['Step 6 - RAG']:.2f}s")
    
    # ========================================================================
    # Step 7: Execute Intermediate Commands (API calls)
    # ========================================================================
    print("\n[Step 7] Executing Select_Row and Select_Column...")
    _t7 = time.perf_counter()
    
    # Merge Select_Row and Select_Column into one batch call
    combined_prompts = []
    combined_methods = []
    combined_indices = []
    
    for method in ['Select_Row', 'Select_Column']:
        if LLM_query_list[method]['query']:
            method_prompts = LLM_query_list[method]['query']
            combined_prompts.extend(method_prompts)
            combined_methods.extend([method] * len(method_prompts))
            combined_indices.extend(range(len(method_prompts)))
            print(f"  Preparing {method}: {len(method_prompts)} prompts")
    
    if combined_prompts:
        print(f"  Executing combined batch: {len(combined_prompts)} total prompts...")
        combined_responses = response_vllm_api(
            combined_prompts,
            sample_num=args.select_sample_num,
            api_base=args.api_base,
            api_key=args.api_key,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            cache_file=os.path.join(args.tmp_save_path, "cache_select_ops.json")
        )

        # Split responses back to respective methods
        response_idx = 0
        for method in ['Select_Row', 'Select_Column']:
            if LLM_query_list[method]['query']:
                method_count = len(LLM_query_list[method]['query'])
                LLM_query_list[method]['response'] = combined_responses[response_idx:response_idx + method_count]
                response_idx += method_count
                print(f"    {method}: {len(LLM_query_list[method]['response'])} responses")
    
    timeline['Step 7 - Select Ops Generation'] = time.perf_counter() - _t7
    print(f"  [Timing] Step 7 - Select Ops Generation: {timeline['Step 7 - Select Ops Generation']:.2f}s")
    
    print("\n[Step 8] Executing Execute_SQL...")
    _t8 = time.perf_counter()
    if LLM_query_list['Execute_SQL']['query']:
        prompt_list = LLM_query_list['Execute_SQL']['query']
        response_list = response_vllm_api(
            prompt_list,
            sample_num=args.sql_sample_num,
            api_base=args.api_base,
            api_key=args.api_key,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            cache_file=os.path.join(args.tmp_save_path, "cache_execute_sql.json")
        )
        LLM_query_list['Execute_SQL']['response'] = response_list
        print(f"  Execute_SQL: {len(response_list)} responses")
    timeline['Step 8 - Execute_SQL Generation'] = time.perf_counter() - _t8
    print(f"  [Timing] Step 8 - Execute_SQL Generation: {timeline['Step 8 - Execute_SQL Generation']:.2f}s")
    
    # ========================================================================
    # Step 9: SQL Parse and Execute
    # ========================================================================
    print("\n[Step 9] Parsing and Executing SQL...")
    _t9 = time.perf_counter()
    
    # Parse Select_Row
    print("  Parsing Select_Row SQL...")
    filtered_tables_row = {}
    row_sql_index_list = LLM_query_list['Select_Row']['index']
    row_sql_response_list = LLM_query_list['Select_Row']['response']
    
    for i in tqdm(range(len(row_sql_index_list)), desc="  Select_Row"):
        index = row_sql_index_list[i]
        sub_table_list = []
        
        for sample_index in range(len(row_sql_response_list[i])):
            original_text = row_sql_response_list[i][sample_index]
            sql = fix_sql_query(
                response_text=original_text,
                table_df=wikitq_df_processed[index],
                table_title=table_titles[index]
            )
            
            try:
                # Hang guard: a model-generated WITH RECURSIVE CTE can spin
                # forever in DuckDB (C-level, no statement timeout) and stall
                # the whole run (observed: table 738 hung 1.5h then killed).
                # WikiTQ table QA never needs recursion, so skip such SQL.
                if sql and 'recursive' in sql.lower():
                    raise ValueError('skip pathological recursive SQL (hang guard)')
                result = executor.sql_exec(
                    sql.replace('``', '`').replace("COUNT(*)", "*"),
                    db, table_id=index
                )
                sub_table_list.append(
                    pd.DataFrame(result['rows'], columns=result['header'])
                )
            except:
                continue
        
        filtered_df = retrieve_rows_by_subtables(
            wikitq_df_processed[index], sub_table_list
        )
        if len(filtered_df) == 0:
            filtered_df = wikitq_df_processed[index]
        filtered_tables_row[index] = filtered_df
    
    # Parse Select_Column
    print("  Parsing Select_Column SQL...")
    filtered_tables = {}
    filtered_headers = {}
    col_sql_index_list = LLM_query_list['Select_Column']['index']
    col_sql_response_list = LLM_query_list['Select_Column']['response']
    
    for i in tqdm(range(len(col_sql_index_list)), desc="  Select_Column"):
        index = col_sql_index_list[i]
        input_df = wikitq_df_processed[index]
        response_list = col_sql_response_list[i]
        
        filtered_table, final_headers = filter_dataframe_from_responses(
            response_list, input_df, add_row_id=True
        )
        filtered_tables[index] = filtered_table
        filtered_headers[index] = final_headers
    
    # Parse Execute_SQL
    print("  Parsing Execute_SQL...")
    sql_exec_df = {}
    sql_executable_count = []
    exec_sql_index_list = LLM_query_list['Execute_SQL']['index']
    exec_sql_response_list = LLM_query_list['Execute_SQL']['response']
    
    for i in tqdm(range(len(exec_sql_index_list)), desc="  Execute_SQL"):
        index = exec_sql_index_list[i]
        sql_exec_df[index] = []
        
        for sample_ind in range(len(exec_sql_response_list[i])):
            original_text = exec_sql_response_list[i][sample_ind]
            sql = fix_sql_query(
                response_text=original_text,
                table_df=wikitq_df_processed[index],
                table_title=table_titles[index]
            )
            
            if sql and 'recursive' in sql.lower():
                # hang guard (see Select_Row site): skip pathological recursive CTE
                df = pd.DataFrame()
            elif sql:
                try:
                    result = executor.sql_exec(
                        sql.replace('``', '`'), db,
                        table_id=index, add_row_id=True
                    )
                    df = pd.DataFrame(result['rows'], columns=result['header'])
                except:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
            
            sql_exec_df[index].append(df)
            
            if len(df) > 0:
                sql_executable_count.append({
                    'id': index,
                    'sample_ind': sample_ind,
                    'sql': sql,
                    'table': df
                })
    
    sql_exec_df_output = merge_clean_and_format_df_dict(sql_exec_df)
    
    # Aggregate processed tables (in memory)
    processed_table = {
        'Base': wikitq_df_processed,
        'Select_Row': filtered_tables_row,
        'Select_Column': filtered_tables,
        'RAG_20_5': RAG_20_5,
        'Execute_SQL': sql_exec_df_output,
        'Execute_SQL_count': sql_executable_count
    }
    
    if args.save_intermediate:
        np.save(f'{args.tmp_save_path}/processed_table.npy', processed_table)
        print(f"  Saved processed tables")
    timeline['Step 9 - SQL Parse & Execute'] = time.perf_counter() - _t9
    print(f"  [Timing] Step 9 - SQL Parse & Execute: {timeline['Step 9 - SQL Parse & Execute']:.2f}s")
    
    # ========================================================================
    # Step 10: Check Model Iteration
    # ========================================================================
    print("\n[Step 10] Running Check Model...")
    _t10 = time.perf_counter()
    
    # Initialize Check Model Data Sequence
    Check_Model_Data_Sequence = {}
    for key in tqdm(ranked_result.keys(), desc="  Initializing check sequences"):
        start_sequence = ranked_result[key]
        Check_Model_Data_Sequence[key] = {
            'id': key,
            'Sequence': start_sequence,
            'Terminated': start_sequence in [['Base'], ['Execute_SQL']],
            'Check_Status': False,
            'Check_Score': 0.0
        }
        
        if not Check_Model_Data_Sequence[key]['Terminated']:
            data_entry = Prepare_Data_for_Operator_Sequence(
                key, start_sequence, dataset, processed_table
            )
            Check_Model_Data_Sequence[key]['data_entry'] = data_entry
    
    # Iterative check (3 loops) - using pre-loaded reranker model
    print("  Running iterative check (3 rounds)...")
    for loop in range(3):
        print(f"    Round {loop + 1}/3...")
        updated_data = batch_rerank_scores(
            reranker_model, Check_Model_Data_Sequence, batch_size=16
        )
        Check_Model_Data_Sequence = updated_data
        
        # Check Terminal Status
        for key in Check_Model_Data_Sequence.keys():
            if Check_Model_Data_Sequence[key]['Terminated']:
                continue
            
            if Check_Model_Data_Sequence[key]['Check_Status']:
                if Check_Model_Data_Sequence[key]['Check_Score'] >= args.check_tau:
                    Check_Model_Data_Sequence[key]['Terminated'] = True
                else:
                    Check_Model_Data_Sequence[key]['Terminated'] = False
                    Check_Model_Data_Sequence[key]['Check_Status'] = False
                    Check_Model_Data_Sequence[key]['Check_Score'] = 0.0
                    
                    current_sequence = Check_Model_Data_Sequence[key]['Sequence']
                    ROLLBACK_seq, terminated_flag = ROLLBACK(current_sequence)
                    Check_Model_Data_Sequence[key]['Sequence'] = ROLLBACK_seq
                    Check_Model_Data_Sequence[key]['Terminated'] = terminated_flag
                    
                    if not terminated_flag:
                        data_entry = Prepare_Data_for_Operator_Sequence(
                            key, ROLLBACK_seq, dataset, processed_table
                        )
                        Check_Model_Data_Sequence[key]['data_entry'] = data_entry
    
    if args.save_intermediate:
        np.save(
            f'{args.tmp_save_path}/Check_Model_Data_Sequence.npy',
            Check_Model_Data_Sequence
        )
    timeline['Step 10 - Check Model'] = time.perf_counter() - _t10
    print(f"  [Timing] Step 10 - Check Model: {timeline['Step 10 - Check Model']:.2f}s")
    
    # ========================================================================
    # Step 11: Add Missing Execute_SQL
    # ========================================================================
    print("\n[Step 11] Adding Missing Execute_SQL...")
    _t11 = time.perf_counter()
    SQL_list_final = []
    for index in range(len(dataset)):
        sequence = Check_Model_Data_Sequence[index]['Sequence']
        if not sequence or 'Execute_SQL' in sequence:
            SQL_list_final.append(index)
    
    add_sql_list = list(
        set(SQL_list_final) - set(LLM_query_list['Execute_SQL']['index'])
    )
    
    if add_sql_list:
        print(f"  Adding {len(add_sql_list)} missing SQL queries...")
        add_sql_query_list = []
        for index in add_sql_list:
            prompt = build_wikitq_prompt_from_df(
                dataset, wikitq_df_processed[index], index,
                template_path='../prompts/sql_reason_fetaqa.txt',
                processed=True
            )
            add_sql_query_list.append(prompt)
        
        add_sql_response_list = response_vllm_api(
            add_sql_query_list,
            sample_num=args.sql_sample_num,
            api_base=args.api_base,
            api_key=args.api_key,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            cache_file=os.path.join(args.tmp_save_path, "cache_additional_sql.json")
        )

        # Parse and execute additional SQL
        sql_exec_df_new = {}
        for i in tqdm(range(len(add_sql_list)), desc="  Additional SQL"):
            index = add_sql_list[i]
            sql_exec_df_new[index] = []
            
            for sample_ind in range(len(add_sql_response_list[i])):
                original_text = add_sql_response_list[i][sample_ind]
                sql = fix_sql_query(
                    response_text=original_text,
                    table_df=wikitq_df_processed[index],
                    table_title=table_titles[index]
                )
                
                if sql:
                    try:
                        result = executor.sql_exec(
                            sql.replace('``', '`'), db,
                            table_id=index, add_row_id=True
                        )
                        df = pd.DataFrame(result['rows'], columns=result['header'])
                    except:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()
                
                sql_exec_df_new[index].append(df)
        
        sql_exec_df_output_new = merge_clean_and_format_df_dict(sql_exec_df_new)
        for index in sql_exec_df_output_new.keys():
            sql_exec_df_output[index] = sql_exec_df_output_new[index]
        
        processed_table['Execute_SQL'] = sql_exec_df_output
    else:
        print("  No missing SQL queries")
    timeline['Step 11 - Add Missing Execute_SQL'] = time.perf_counter() - _t11
    print(f"  [Timing] Step 11 - Add Missing Execute_SQL: {timeline['Step 11 - Add Missing Execute_SQL']:.2f}s")
    
    # ========================================================================
    # Step 12: Generate Final QA Prompts
    # ========================================================================
    print("\n[Step 12] Generating Final QA Prompts...")
    _t12 = time.perf_counter()
    prompt_list = []
    for index in tqdm(range(len(dataset)), desc="  Building QA prompts"):
        sequence = Check_Model_Data_Sequence[index]['Sequence']
        
        # Ensure data_entry exists
        if 'data_entry' not in Check_Model_Data_Sequence[index]:
            try:
                data_entry = Prepare_Data_for_Operator_Sequence(
                    index, sequence, dataset, processed_table
                )
                Check_Model_Data_Sequence[index]['data_entry'] = data_entry
            except Exception:
                Check_Model_Data_Sequence[index]['data_entry'] = {
                    'table': processed_table['Base'][index],
                    'SQL': ''
                }
        
        prompt = build_wikitq_prompt_from_df(
            dataset,
            Check_Model_Data_Sequence[index]['data_entry']['table'],
            index,
            template_path='../prompts/text_reason_fetaqa.txt',
            processed=True
        )
        
        if not sequence or 'Execute_SQL' in sequence:
            evidence = table_to_str_sql(processed_table['Execute_SQL'][index])
            prompt = prompt + evidence
        
        prompt_list.append(prompt)
    timeline['Step 12 - Build QA Prompts'] = time.perf_counter() - _t12
    print(f"  [Timing] Step 12 - Build QA Prompts: {timeline['Step 12 - Build QA Prompts']:.2f}s")
    
    # ========================================================================
    # Step 13: Execute Final QA and Evaluate
    # ========================================================================
    print("\n[Step 13] Executing Final QA...")
    _t13a = time.perf_counter()
    qa_final = response_vllm_api(
        prompt_list,
        sample_num=1,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        temperature=0,
        top_p=1,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        cache_file=os.path.join(args.tmp_save_path, "cache_final_qa.json")
    )

    # Create result dataframe
    wikitq_df = pd.DataFrame(dataset)
    wikitq_df['instruction'] = prompt_list
    wikitq_df['predict'] = [str(s[0]) if isinstance(s, list) else str(s) 
                            for s in qa_final]
    
    # Save results
    wikitq_df.to_csv(f'{args.tmp_save_path}/final_results.csv', index=False)
    print(f"  Saved results to {args.tmp_save_path}/final_results.csv")
    timeline['Step 13a - Final QA Generation'] = time.perf_counter() - _t13a
    print(f"  [Timing] Step 13a - Final QA Generation: {timeline['Step 13a - Final QA Generation']:.2f}s")
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Evaluate — FetaQA is free-form QA, scored by ROUGE-L fmeasure (conference
    # metric), NOT exact match. Extraction mirrors the conference fetaqa_score.py
    # ('Answer: ' marker) with a 'Final Answer:' fallback.
    print("\n" + "="*80)
    print("EVALUATION (FetaQA ROUGE-L)")
    print("="*80)
    _t13b = time.perf_counter()
    import re as _re
    from rouge_score import rouge_scorer as _rs

    def _extract_feta_answer(raw):
        s = str(raw)
        if 'Answer: ' in s:
            s = s.split('Answer: ', 1)[1]
        else:
            marks = list(_re.finditer(
                r'(?:final\s+answer|the\s+answer\s+is|answer)\s*[:：]\s*',
                s, _re.IGNORECASE))
            if marks:
                s = s[marks[-1].end():]
        s = s.replace("\n```", "")
        s = _re.sub(r'^\s*therefore,?\s*the answer is\s*[:：]?\s*', '', s,
                    flags=_re.IGNORECASE)
        return s.strip().strip('"\'').strip()

    _scorer = _rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    _r1 = _r2 = _rl = 0.0
    _n = 0
    for _i in range(len(dataset)):
        _gold = str(dataset[_i].get('answer', '')).strip()
        if not _gold:
            continue
        _pred = _extract_feta_answer(wikitq_df['predict'].iloc[_i]) or 'unknown answer'
        try:
            _sc = _scorer.score(_gold, _pred)
        except Exception:
            continue
        _r1 += _sc['rouge1'].fmeasure
        _r2 += _sc['rouge2'].fmeasure
        _rl += _sc['rougeL'].fmeasure
        _n += 1
    acc_all = (_rl / _n) if _n else 0.0

    with open(f'{args.tmp_save_path}/evaluation_results.json', 'w') as f:
        json.dump({
            'metric': 'rougeL_fmeasure',
            'rougeL_f': acc_all,
            'rouge1_f': (_r1 / _n) if _n else 0.0,
            'rouge2_f': (_r2 / _n) if _n else 0.0,
            'n_scored': _n,
            'total_samples': len(dataset),
        }, f, indent=2)
    print(f"  FetaQA ROUGE-L fmeasure = {acc_all:.4f}  (n={_n})")
    timeline['Step 13b - Evaluation'] = time.perf_counter() - _t13b
    print(f"  [Timing] Step 13b - Evaluation: {timeline['Step 13b - Evaluation']:.2f}s")

    # Timeline summary
    print("\n" + "="*80)
    print("TIMELINE SUMMARY")
    print("="*80)
    name_width = max((len(k) for k in timeline.keys()), default=0)
    for name, secs in timeline.items():
        print(f"{name.ljust(name_width)} : {secs:.2f}s")
    
    print(f"\n{'='*80}")
    print(f"Pipeline completed successfully!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"FetaQA ROUGE-L fmeasure: {acc_all:.4f}")
    print(f"Results saved to: {args.tmp_save_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

