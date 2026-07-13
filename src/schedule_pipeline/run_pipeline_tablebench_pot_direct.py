#!/usr/bin/env python3
"""
Enhanced TableBench PoT Pipeline with Iterative Retry and Direct Answer
- Iterative retry: if Python execution fails, append error and retry (max 3 iterations)
- **Direct Answer Mode**: If Python succeeds, use execution result directly as answer (no LLM QA step)
- **Fallback Mode**: If Python fails, use LLM to generate answer from table
- Reports execution success rate and direct answer usage rate
- Sample 3 Python codes per question
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List
import re
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.async_llm import infer_prompts
from utils.schedule_utils import table_to_str_sql
from utils.evaluator import evaluate_tablebench_predictions
from utils.python_executor import execute_python_code

import multiprocessing as mp

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced TableBench PoT Pipeline with Direct Answer")
    
    parser.add_argument('--llm_path', type=str, 
                       default='/data/workspace/yanmy/models/Qwen2.5-7B-Instruct/', help='Path to LLM model')
    parser.add_argument('--llm_name', type=str, 
                       default='qwen3-4b', help='Model name registered in vLLM API server')
    parser.add_argument('--dataset_name', type=str, default='tablebench', help='Dataset name')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--tmp_save_path', type=str,
                       default='datasets/schedule_test/tablebench_pot_direct',
                       help='Temporary save path')
    parser.add_argument('--tablebench_jsonl_path', type=str,
                       default='../datasets/TableBench/TableBench_PoT.jsonl',
                       help='Path to TableBench PoT JSONL file')
    
    parser.add_argument('--n_parallel', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--llm_concurrency', type=int, default=32, help='Max concurrent requests')
    
    parser.add_argument('--code_sample_num', type=int, default=3, help='Samples for Python code generation')
    parser.add_argument('--max_iterations', type=int, default=3, help='Max retry iterations for failed Python execution')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8, help='Sampling top_p')
    
    parser.add_argument('--first_n', type=int, default=-1, help='Only process first N samples')
    parser.add_argument('--random_sample', action='store_true', help='Randomly sample instead of first N')
    parser.add_argument('--use_api', action='store_true', help='Use async API')
    
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1", help='vLLM API Base URL')
    parser.add_argument('--api_key', type=str, default="api-key-qwen3", help='vLLM API Key')
    
    return parser.parse_args()

def load_tablebench_dataset(jsonl_path: str, first_n: int = -1, random_sample: bool = False) -> List[Dict]:
    print(f"Loading TableBench dataset from {jsonl_path}...")
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
    
    print(f"Loaded {len(data)} total samples")
    
    if first_n > 0:
        if random_sample:
            import random
            print(f"Randomly sampling {first_n} samples...")
            data = random.sample(data, min(first_n, len(data)))
        else:
            print(f"Taking first {first_n} samples...")
            data = data[:first_n]
    
    print(f"Using {len(data)} samples for testing")
    return data

def make_unique_columns(columns: List[str]) -> List[str]:
    seen = {}
    result = []
    for col in columns:
        col_str = str(col).strip() if col else "unnamed"
        if not col_str: col_str = "unnamed"
        col_str = col_str.replace('\n', ' ').replace('\r', ' ')
        col_str = re.sub(r'\s+', ' ', col_str).strip()
        if col_str in seen:
            seen[col_str] += 1
            result.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            result.append(col_str)
    return result

def tablebench_table_to_df(item: Dict) -> pd.DataFrame:
    columns = item['table']['columns']
    data = item['table']['data']
    unique_columns = make_unique_columns(columns)
    return pd.DataFrame(data, columns=unique_columns)

def build_pot_prompt_with_error(item: Dict, df: pd.DataFrame, template_path: str,
                                 error_msg: str = None, iteration: int = 0) -> str:
    """Build PoT prompt with optional error feedback for retry."""
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    table_dict = {
        'columns': df.columns.tolist(),
        'data': df.values.tolist()
    }
    table_str = str(table_dict)
    
    question = item.get('question', '')
    
    # Add error feedback for retry iterations
    error_feedback = ""
    if error_msg and iteration > 0:
        error_feedback = f"""

[Previous Attempt Failed - Iteration {iteration}]
Error: {error_msg}
Please generate corrected Python code that avoids this error.
"""
    
    prompt = template.strip() + f"\n\nRead the table below in JSON format:\n[TABLE] \n{table_str}\n\nLet's get start!\nQuestion: {question}{error_feedback}"
    return prompt

def extract_python_code(response: str) -> str:
    """Enhanced code extraction with better parsing logic."""
    # Try to find code block with python tag
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find any code block
    match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Check if it looks like Python code
        if any(keyword in code for keyword in ['import', 'def ', 'print', 'pd.', 'df']):
            return code
    
    # Try to find code after "Step" markers
    if 'Step' in response and '```' not in response:
        lines_resp = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines_resp:
            if line.strip().startswith('import ') or line.strip().startswith('df '):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()
    
    # Check if response contains Python-like code without markers
    if any(keyword in response for keyword in ['import pandas', 'pd.read_csv', 'df =', 'print(']):
        lines_resp = response.split('\n')
        code_lines = []
        for line in lines_resp:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('Step'):
                if any(kw in stripped for kw in ['import', 'pd.', 'df', 'print', '=', 'mean(', 'sum(']):
                    code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()
    
    return ""

def extract_direct_answer_from_execution(execution_result: str) -> str:
    """
    Extract direct answer from Python execution result.
    Returns the execution output as the answer directly.
    """
    if not execution_result or "Execution Error" in execution_result:
        return ""
    
    # Clean up the result
    result = execution_result.strip()
    
    # Remove common prefixes
    for prefix in ['Answer:', 'Result:', 'Output:']:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
    
    # Limit length
    result = result[:200]
    
    return result

def batch_execute_python_with_retry(raw_data: List[Dict], 
                                    processed_dfs: Dict[int, pd.DataFrame],
                                    args, template_path: str, code_dir: str) -> tuple:
    """
    Batch execute Python with iterative retry logic.
    Returns: (all_stats, execution_stats)
    """
    # Initialize stats for all samples
    all_stats = []
    
    for idx, item in enumerate(raw_data):
        all_stats.append({
            'idx': idx,
            'question': item.get('question', ''),
            'iterations': [],
            'final_success': False,
            'final_result': None,
            'direct_answer': None,
            'total_attempts': 0,
            'successful_attempts': 0
        })
    
    # Track which samples need processing in each iteration
    active_samples = set(range(len(raw_data)))
    
    for iteration in range(args.max_iterations):
        if not active_samples:
            break
        
        print(f"\n  Iteration {iteration}: Processing {len(active_samples)} samples...")
        
        # Build prompts for all active samples
        prompt_list = []
        idx_mapping = []
        
        for idx in sorted(active_samples):
            item = raw_data[idx]
            df = processed_dfs[idx]
            stats = all_stats[idx]
            
            # Build prompt with error feedback if retry
            error_msg = None
            if iteration > 0 and stats['iterations']:
                last_iter = stats['iterations'][-1]
                if last_iter['attempts']:
                    error_msg = last_iter['attempts'][0].get('error', 'Python execution failed')
            
            prompt = build_pot_prompt_with_error(item, df, template_path, error_msg, iteration)
            prompt_list.append(prompt)
            idx_mapping.append(idx)
        
        # Batch generate Python code
        print(f"  Generating Python code for {len(prompt_list)} samples...")
        responses, _, _ = infer_prompts(
            prompt_list,
            sample_num=args.code_sample_num,
            temperature=args.temperature,
            top_p=args.top_p,
            llm_name=args.llm_name,
            api_base=args.api_base,
            api_key=args.api_key,
            concurrency=args.llm_concurrency
        )
        
        # Process responses for each sample
        newly_succeeded = set()
        
        for prompt_idx, response_list in enumerate(responses):
            idx = idx_mapping[prompt_idx]
            item = raw_data[idx]
            df = processed_dfs[idx]
            stats = all_stats[idx]
            
            iteration_stats = {
                'iteration': iteration,
                'attempts': [],
                'success': False
            }
            
            # Try each generated code
            for sample_idx, response_text in enumerate(response_list):
                stats['total_attempts'] += 1
                attempt = {
                    'sample_idx': sample_idx,
                    'response': response_text,
                    'code': None,
                    'success': False,
                    'error': None,
                    'result': None
                }
                
                # Save response
                response_file = os.path.join(code_dir, f"sample_{idx}_iter_{iteration}_attempt_{sample_idx}_response.txt")
                with open(response_file, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                
                # Parse code
                code = extract_python_code(response_text)
                attempt['code'] = code
                
                if not code or len(code.strip()) < 10:
                    attempt['error'] = "Failed to parse Python code from response"
                    iteration_stats['attempts'].append(attempt)
                    continue
                
                # Save code
                code_file = os.path.join(code_dir, f"sample_{idx}_iter_{iteration}_attempt_{sample_idx}.py")
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                # Execute code
                output = execute_python_code(code, df)
                
                if output and "Execution Error" not in output:
                    attempt['success'] = True
                    attempt['result'] = output.strip()
                    iteration_stats['success'] = True
                    stats['successful_attempts'] += 1
                    stats['final_success'] = True
                    stats['final_result'] = output.strip()
                    # Extract direct answer
                    stats['direct_answer'] = extract_direct_answer_from_execution(output.strip())
                else:
                    attempt['error'] = output if output else "Execution failed with no output"
                
                iteration_stats['attempts'].append(attempt)
            
            stats['iterations'].append(iteration_stats)
            
            # Mark as succeeded if any attempt worked
            if iteration_stats['success']:
                newly_succeeded.add(idx)
        
        # Remove succeeded samples from active set
        active_samples -= newly_succeeded
        print(f"  Iteration {iteration}: {len(newly_succeeded)} samples succeeded, {len(active_samples)} remaining")
    
    # Calculate statistics
    total_samples = len(all_stats)
    total_attempts = sum(s['total_attempts'] for s in all_stats)
    successful_attempts = sum(s['successful_attempts'] for s in all_stats)
    final_success_count = sum(1 for s in all_stats if s['final_success'])
    direct_answer_count = sum(1 for s in all_stats if s['direct_answer'] is not None)
    
    execution_stats = {
        'total_samples': total_samples,
        'total_attempts': total_attempts,
        'successful_attempts': successful_attempts,
        'final_success_count': final_success_count,
        'direct_answer_count': direct_answer_count,
        'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
        'sample_success_rate': final_success_count / total_samples if total_samples > 0 else 0,
        'direct_answer_rate': direct_answer_count / total_samples if total_samples > 0 else 0
    }
    
    return all_stats, execution_stats


def main():
    args = parse_args()
    
    # Add timestamp to save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.tmp_save_path = f"{args.tmp_save_path}_{timestamp}"
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    print("="*80)
    print("TableBench PoT Pipeline - Direct Answer Version")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Save Path: {args.tmp_save_path}")
    print(f"Dataset: {args.tablebench_jsonl_path}")
    print(f"First N: {args.first_n}")
    print(f"Random Sample: {args.random_sample}")
    print(f"LLM Name: {args.llm_name}")
    print(f"API Base: {args.api_base}")
    print(f"Code Sample Num: {args.code_sample_num}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Temperature: {args.temperature}")
    print(f"Top P: {args.top_p}")
    print(f"Concurrency: {args.llm_concurrency}")
    print(f"Mode: Direct Answer (Python result) with LLM Fallback")
    print("="*80)
    print()
    
    overall_start = time.perf_counter()
    
    print("[Step 1] Loading Data...")
    raw_data = load_tablebench_dataset(args.tablebench_jsonl_path, args.first_n, args.random_sample)
    
    print("\n[Step 2] Preprocessing tables...")
    processed_dfs = {}
    for idx, item in enumerate(tqdm(raw_data)):
        processed_dfs[idx] = tablebench_table_to_df(item)
    
    print("\n[Step 3] Executing Python with Iterative Retry...")
    code_dir = os.path.join(args.tmp_save_path, "generated_codes")
    os.makedirs(code_dir, exist_ok=True)
    
    template_path = os.path.join(os.path.dirname(__file__), '../prompts/python_reason_tablebench.txt')
    
    all_stats, execution_stats = batch_execute_python_with_retry(
        raw_data,
        processed_dfs,
        args,
        template_path,
        code_dir
    )
    
    # Save detailed stats
    stats_file = os.path.join(args.tmp_save_path, 'execution_stats_detailed.json')
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    # Save summary
    summary_file = os.path.join(args.tmp_save_path, 'execution_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(execution_stats, f, indent=2)
    
    print(f"\n--- Execution Statistics ---")
    print(f"Total samples: {execution_stats['total_samples']}")
    print(f"Total attempts: {execution_stats['total_attempts']}")
    print(f"Successful: {execution_stats['successful_attempts']} ({execution_stats['success_rate']*100:.1f}%)")
    print(f"Sample success rate: {execution_stats['sample_success_rate']*100:.1f}%")
    print(f"Direct answer count: {execution_stats['direct_answer_count']} ({execution_stats['direct_answer_rate']*100:.1f}%)")
    print(f"----------------------------\n")
    
    print("\n[Step 4] Generating Final Answers...")
    print(f"  Direct answers from Python: {execution_stats['direct_answer_count']}")
    print(f"  Need LLM fallback: {execution_stats['total_samples'] - execution_stats['direct_answer_count']}")
    
    # Identify samples that need LLM fallback
    need_llm_indices = [idx for idx, s in enumerate(all_stats) if not s['final_success']]
    
    # Initialize predictions with direct answers
    preds = [s['direct_answer'] if s['direct_answer'] is not None else None for s in all_stats]
    answer_sources = ['direct_python' if s['direct_answer'] is not None else 'llm_fallback' for s in all_stats]
    
    # Generate LLM answers for failed Python cases
    if need_llm_indices:
        print(f"\n  Generating LLM fallback answers for {len(need_llm_indices)} samples...")
        qa_template_path = os.path.join(os.path.dirname(__file__), '../prompts/text_reason_wtq_nocase.txt')
        
        with open(qa_template_path, 'r', encoding='utf-8') as f:
            qa_template = f.read()
        
        prompt_list = []
        for idx in need_llm_indices:
            item = raw_data[idx]
            df = processed_dfs[idx]
            
            temp_df = df.copy()
            if 'row_id' not in temp_df.columns:
                temp_df.insert(0, 'row_id', temp_df.index)
            table_str = temp_df.to_string(index=False)
            
            columns = df.columns.tolist()
            all_cols = ['row_id'] + columns if 'row_id' not in columns else columns
            
            table_title = item.get('id', 'Table')
            table_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(table_title))
            
            schema_cols = [f"`{col}` text" for col in columns]
            if 'row_id' not in columns:
                schema_cols.insert(0, "row_id int")
            col_defs = ",\n\t".join(schema_cols)
            schema = f"CREATE TABLE {table_name}(\n\t{col_defs})"
            
            question = item.get('question', '')
            
            input_section = f"""
<input>
{schema}
/*
SELECT * FROM w;
{table_str}
*/
columns: {all_cols}
Q: {question}
<output>"""
            
            prompt = qa_template.strip() + "\n" + input_section
            prompt_list.append(prompt)
        
        # Batch generate QA responses
        qa_responses, _, _ = infer_prompts(
            prompt_list,
            sample_num=1,
            temperature=0,
            top_p=1,
            llm_name=args.llm_name,
            api_base=args.api_base,
            api_key=args.api_key,
            concurrency=args.llm_concurrency
        )
        
        # Fill in LLM predictions
        for i, idx in enumerate(need_llm_indices):
            qa = qa_responses[i]
            pred_str = qa[0] if isinstance(qa, list) else str(qa)
            # Extract "The answer is:" pattern
            match = re.search(r'(?:the answer is|therefore|answer):\s*(.+)', pred_str, re.IGNORECASE)
            if match:
                pred_str = match.group(1).strip()
            pred_str = pred_str.strip().strip('"\'')[:200]
            preds[idx] = pred_str
    
    print("\n[Step 5] Evaluation...")
    
    golds = [str(item.get('answer', '')) for item in raw_data]
    
    eval_results = evaluate_tablebench_predictions(preds, golds)
    
    # Save results
    results_df = pd.DataFrame({
        'id': [item.get('id', idx) for idx, item in enumerate(raw_data)],
        'question': [item.get('question', '') for item in raw_data],
        'gold_answer': golds,
        'prediction': preds,
        'answer_source': answer_sources,
        'python_success': [s['final_success'] for s in all_stats],
        'total_attempts': [s['total_attempts'] for s in all_stats],
        'successful_attempts': [s['successful_attempts'] for s in all_stats],
        'iterations_used': [len(s['iterations']) for s in all_stats]
    })
    results_df.to_csv(os.path.join(args.tmp_save_path, 'results.csv'), index=False)
    
    with open(os.path.join(args.tmp_save_path, 'evaluation.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    total_time = time.perf_counter() - overall_start
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total Samples: {execution_stats['total_samples']}")
    print(f"Total Python Attempts: {execution_stats['total_attempts']}")
    print(f"Successful Python Executions: {execution_stats['successful_attempts']}")
    print(f"Python Execution Success Rate: {execution_stats['success_rate']*100:.2f}%")
    print(f"Sample Success Rate: {execution_stats['sample_success_rate']*100:.2f}%")
    print()
    print("Answer Sources:")
    print(f"  Direct Python Answer: {execution_stats['direct_answer_count']} ({execution_stats['direct_answer_rate']*100:.2f}%)")
    print(f"  LLM Fallback: {execution_stats['total_samples'] - execution_stats['direct_answer_count']} ({(1-execution_stats['direct_answer_rate'])*100:.2f}%)")
    print()
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Average ROUGE-L: {eval_results['avg_rouge_l']:.4f}")
    print(f"Accuracy@0.5: {eval_results['accuracy_at_0.5']*100:.2f}%")
    print(f"Accuracy@0.8: {eval_results['accuracy_at_0.8']*100:.2f}%")
    print()
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {args.tmp_save_path}")
    print("="*80)


if __name__ == "__main__":
    main()
