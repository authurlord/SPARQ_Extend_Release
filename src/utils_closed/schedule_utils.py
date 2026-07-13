import pickle
import numpy as np
import pandas as pd
import os
import json
from utils.utils import load_data_split
from utils.evaluator import Evaluator
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
from FlagEmbedding import FlagReranker
from typing import Dict, List, Any

from datasets import load_dataset

def table_to_str(new_df):
    string = '\nHere is an additional evidence to help the answering process.\nAdditional Evidence:\n/*\n'
    string += 'col : ' + ' | '.join(new_df.columns) + '\n'
    for row_id, row in new_df.iloc[:len(new_df)].iterrows():
        string += f'row {row_id} : '
        for column_id, header in enumerate(new_df.columns):
            string += str(row[header])
            if column_id != len(new_df.columns) - 1:
                string += ' | '
        string += '\n'
    string += '*/\n'
    return string

def table_to_str_sql(new_df):
    if len(new_df)>0:
        string = '\nHere is an additional evidence to help the answering process.\nAdditional Evidence:\n/*\n'
        string += 'col : ' + ' | '.join(new_df.columns) + '\n'
        for row_id, row in new_df.iloc[:len(new_df)].iterrows():
            string += f'row {row_id} : '
            for column_id, header in enumerate(new_df.columns):
                string += str(row[header])
                if column_id != len(new_df.columns) - 1:
                    string += ' | '
            string += '\n'
        string += '*/\n'
    else:
        string = ''
    return string

def find_intersection_and_add_row_id(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    根据索引和公共列找到两个 DataFrame 的交集，并返回一个新 DataFrame，
    其中保留了交集行的原始索引，并在第一列添加了新的 row_id。

    参数:
    df1 (pd.DataFrame): 第一个 DataFrame。
    df2 (pd.DataFrame): 第二个 DataFrame。

    返回:
    pd.DataFrame: 包含两个 DataFrame 交集的 DataFrame。
    """
    # 1. 忽略可能存在的 'row_id' 列
    df1_clean = df1.drop(columns=['row_id'], errors='ignore')
    df2_clean = df2.drop(columns=['row_id'], errors='ignore')

    # 2. 找到两个 DataFrame 的公共列
    common_cols = list(set(df1_clean.columns) & set(df2_clean.columns))
    
    # 3. 找到两个 DataFrame 的公共索引
    # 使用 pd.Index.intersection() 方法找到两个索引的交集
    common_indices = df1_clean.index.intersection(df2_clean.index)
    
    # 4. 根据公共索引和公共列提取交集数据
    # .loc[] 方法可以同时按行（索引）和列进行选择
    intersected_df = df1_clean.loc[common_indices, common_cols]

    # 5. 在第一列 (索引 0) 插入新的 'row_id' 列
    # 使用 range(1, len(common_indices) + 1) 生成从1开始的序列
    intersected_df.insert(0, 'row_id', range(1, len(common_indices) + 1))
    
    return intersected_df

def Prepare_Data_for_Operator_Sequence(index,sequence,dataset, processed_table):
    data_entry = {
    'id': dataset[index]['id'],
    'query': dataset[index]['question'], ## 没有ReWrite，暂时不做变化
    # 'table': wikitq_df_processed[index],
    'title': dataset[index]['table']['page_title'], ## 不变
    # 'SQL': sql_command['sql'] + table_to_str(sql_command['table']) ,
        }
    if sequence.__contains__('Execute_SQL'): ## 有Execute_SQL计算符
        sql_list = [sql_count for sql_count in processed_table['Execute_SQL_count'] if sql_count['id']==index]
        # sql_command = sql_list[0]
        if len(sql_list)==0:
            SQL = ''
        else:
            sql_command = sql_list[0]
            SQL = sql_command['sql'] + table_to_str(sql_command['table'])
        data_entry['SQL'] = SQL
    else:
        data_entry['SQL'] = ''
    extraction_sequence = [s for s in sequence if s!='Execute_SQL']
    # print(extraction_sequence)
    if len(extraction_sequence)==1: ## Only One Operator
        method = extraction_sequence[0]
        data_entry['table'] = processed_table[method][index]
    elif len(extraction_sequence)==2:
        table_1 = processed_table[extraction_sequence[0]][index]
        table_2 = processed_table[extraction_sequence[1]][index]
        # print(table_1.shape, table_2.shape)
        table_intersection = find_intersection_and_add_row_id(table_1,table_2)
        data_entry['table'] = table_intersection
    elif len(extraction_sequence)==0:
        data_entry['table'] = processed_table['Base'][index]
    assert data_entry.__contains__('SQL')
    assert data_entry.__contains__('table')
    return data_entry

def format_document(title: str, table_df: pd.DataFrame, sql: str) -> str:
    """
    Constructs a single document string from the title, table, and SQL query.
    This must be identical to the function used in the training data preparation.
    
    Args:
        title (str): The title of the table.
        table_df (pd.DataFrame): The table data.
        sql (str): The SQL query, which can be None or empty.

    Returns:
        str: A single formatted string representing the document.
    """
    # The order is changed to Title, SQL, Table to prevent SQL from being truncated.
    doc = f"Title: {title}"
    if sql and pd.notna(sql):
        doc += f"\nSQL:\n{sql}"
    
    table_string = table_df.to_string(index=False)
    doc += f"\nTable:\n{table_string}"
    return doc

def batch_rerank_scores(
    reranker: FlagReranker, 
    data_dict: Dict[int, Dict[str, Any]], 
    batch_size: int = 256
) -> Dict[int, Dict[str, Any]]:
    """
    Computes scores for a dictionary of data entries using a FlagReranker model,
    only processing items where 'Terminated' is False.

    Args:
        reranker (FlagReranker): An initialized FlagReranker model instance.
        data_dict (Dict[int, Dict[str, Any]]): A dictionary where each value is another
            dictionary with keys like 'id', 'Terminated', 'Check_Status', 'Check_Score',
            and optionally 'data_entry'.
        batch_size (int): The batch size for inference.

    Returns:
        Dict[int, Dict[str, Any]]: The input dictionary with 'Check_Status' and 'Check_Score'
            updated for the processed entries.
    """
    if not data_dict:
        return {}

    sentence_pairs = []
    keys_to_update = []

    print("Filtering items to rerank (Terminated=False and data_entry exists)...")
    for key, item in data_dict.items():
        if not item.get('Terminated', True) and 'data_entry' in item:
            item_data = item['data_entry']
            query = item_data['query']
            document = format_document(
                item_data['title'],
                item_data['table'],
                item_data['SQL']
            )
            sentence_pairs.append([query, document])
            keys_to_update.append(key)
    
    if not sentence_pairs:
        print("No items to process.")
        return data_dict

    print(f"Computing scores for {len(sentence_pairs)} items with batch size {batch_size}...")
    scores = reranker.compute_score(
        sentence_pairs, 
        batch_size=batch_size,
        normalize=True
    )

    # Normalize scores to handle scalar outputs for single-item batches
    try:
        scores_array = np.array(scores)
        if scores_array.ndim == 0:
            scores_list = [float(scores_array)]
        else:
            scores_list = scores_array.reshape(-1).tolist()
    except Exception:
        # Fallbacks for unexpected types
        try:
            scores_list = [float(scores)]
        except Exception:
            scores_list = list(scores) if isinstance(scores, (list, tuple)) else [scores]

    print("Updating scores in the original dictionary...")
    limit = min(len(scores_list), len(keys_to_update))
    if limit < len(keys_to_update):
        print(f"[WARN] scores/items mismatch: scores={len(scores_list)} items={len(keys_to_update)}")
    for i in range(limit):
        key = keys_to_update[i]
        data_dict[key]['Check_Status'] = True
        data_dict[key]['Check_Score'] = scores_list[i]
        
    print("Reranking complete.")
    return data_dict

def ROLLBACK(current_sequence):
    terminated_flag = False
    ROLLBACK_seq = []
    if len(current_sequence)==0: ## 空，触发 FORWARD
        ROLLBACK_seq =  ['Execute_SQL']
        terminated_flag = True
    elif len(current_sequence)==1: ## 只有一个操作符
        if current_sequence[0]=='Execute_SQL' or current_sequence[0]=='Base':
            ROLLBACK_seq =  ['Execute_SQL']
            terminated_flag = True
        else:
            ROLLBACK_seq =  ['Base'] ## 检查 Base是否满足条件，不满足则回退到上一条
            terminated_flag = False
    elif len(current_sequence)>=2:
        extraction_sequence = [s for s in current_sequence if s!='Execute_SQL']
        ROLLBACK_seq = extraction_sequence[:-1]
        if current_sequence.__contains__('Execute_SQL'):
            ROLLBACK_seq.append('Execute_SQL')
        if ROLLBACK_seq == ['Execute_SQL']:
            terminated_flag = True
    return ROLLBACK_seq, terminated_flag

def merge_clean_and_format_df_dict(data_dict: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Merges DataFrames within a dictionary's lists, deduplicates, infers numeric types,
    and fills NaN values with an empty string.

    Args:
        data_dict (Dict[str, List[pd.DataFrame]]): 
            A dictionary where each value is a list of n DataFrames.
            These DataFrames can be empty.

    Returns:
        Dict[str, pd.DataFrame]: 
            A processed dictionary with merged, cleaned, and formatted DataFrames.
    """
    processed_dict = {}

    for key, df_list in data_dict.items():
        # 1. Filter out invalid or empty DataFrames
        non_empty_dfs = [df for df in df_list if isinstance(df, pd.DataFrame) and not df.empty]

        if not non_empty_dfs:
            processed_dict[key] = pd.DataFrame()
            continue

        # 2. Concatenate all DataFrames and drop duplicates
        try:
            combined_df = pd.concat(non_empty_dfs, ignore_index=True)
        except:
            # combined_df = pd.concat(non_empty_dfs, ignore_index=False)
            print(non_empty_dfs)
            combined_df = non_empty_dfs[0]
        unique_df = combined_df.drop_duplicates().reset_index(drop=True)

        # 3. Infer data types for each column
        for col in unique_df.columns:
            series_no_na = unique_df[col].dropna()
            
            if series_no_na.empty:
                continue
            try:
                numeric_series = pd.to_numeric(series_no_na, errors='coerce')
            except:
                print(numeric_series)
                continue
            if numeric_series.isna().all():
                continue
            
            # Check if all non-NaN values are integers (no decimal parts)
            is_all_integer = (numeric_series.dropna() == numeric_series.dropna().round()).all()

            if is_all_integer:
                # Use nullable integer type 'Int64' to handle potential NaNs during conversion
                # First convert to float, then round, then cast to Int64 to avoid casting errors
                try:
                    float_col = pd.to_numeric(unique_df[col], errors='coerce')
                    unique_df[col] = float_col.round().astype('Int64')
                except (TypeError, ValueError):
                    # If Int64 conversion fails, fall back to float
                    unique_df[col] = pd.to_numeric(unique_df[col], errors='coerce').astype(float)
            else:
                unique_df[col] = pd.to_numeric(unique_df[col], errors='coerce').astype(float)
        
        # 4. **FIX:** Convert any 'Int64' columns to 'object' type before filling with a string.
        # This prevents the TypeError because 'object' dtype can hold mixed types (numbers and strings).
        for col in unique_df.select_dtypes(include=['Int64']).columns:
            unique_df[col] = unique_df[col].astype('object')

        # 5. Now, safely fill all remaining NaN / <NA> values with an empty string ''
        final_df = unique_df.fillna('')
        
        processed_dict[key] = final_df
        
    return processed_dict

def retrieve_rows_by_subtables(
    large_table: pd.DataFrame,
    sub_tables: List[pd.DataFrame],
    row_id_col: str = "row_id",
    keep_original_if_no_match: bool = True
) -> pd.DataFrame:
    """
    从 large_table 中检索出所有在 sub_tables 列表中出现的 row_id 对应的行。
    (Version 2: Now handles and removes duplicate columns from sub_tables)
    
    参数：
        large_table: 待检索的大表 (pd.DataFrame)。row_id 可能是列也可能是索引。
        sub_tables: 包含若干 sub_table 的列表，每个 sub_table 包含 row_id 列（或 index 为 row_id）。
        row_id_col: row_id 的列名（默认 "row_id"）。
        keep_original_if_no_match: 如果所有 row_id 都没在 large_table 中匹配到，是否返回原始 large_table（True），
                                   否则返回空 DataFrame（False）。
    返回：
        匹配到的 large_table 的行（pd.DataFrame 副本）。如果没有匹配，根据 keep_original_if_no_match 返回原表或空表。
    """
    all_row_ids = set()
    for st_original in sub_tables:
        if st_original is None or st_original.empty:
            continue

        # =========================================================================
        # == MODIFICATION: Check for and remove duplicate columns =================
        # =========================================================================
        # De-duplicate columns to prevent errors if `row_id_col` is duplicated.
        # This keeps the first occurrence of each column name.
        st = st_original.loc[:, ~st_original.columns.duplicated()]
        # =========================================================================

        ids_to_add = None
        if row_id_col in st.columns:
            # Prioritize the column if it exists
            ids_to_add = st[row_id_col]
        else:
            # Fallback to using the index
            ids_to_add = pd.Series(st.index)
        
        # Add the collected, cleaned IDs to the main set
        all_row_ids.update(ids_to_add.dropna().unique())

    if not all_row_ids:
        # No usable row_ids were found in any sub_tables
        return large_table.copy() if keep_original_if_no_match else pd.DataFrame(columns=large_table.columns)

    # Convert row_ids to the same type as the column/index for safe matching
    target_dtype = None
    if row_id_col in large_table.columns:
        target_dtype = large_table[row_id_col].dtype
    else:
        target_dtype = large_table.index.dtype
    
    try:
        # Ensure consistent types for comparison
        typed_row_ids = {pd.Series(list(all_row_ids), dtype=target_dtype).iloc[i] for i in range(len(all_row_ids))}
    except (TypeError, ValueError):
        # If type conversion fails, proceed with the original set
        typed_row_ids = all_row_ids

    # Find matches in the large_table
    if row_id_col in large_table.columns:
        mask = large_table[row_id_col].isin(typed_row_ids)
        result = large_table.loc[mask]
    else:
        mask = large_table.index.isin(typed_row_ids)
        result = large_table.loc[mask]

    if result.empty:
        return large_table.copy() if keep_original_if_no_match else pd.DataFrame(columns=large_table.columns)

    # Return a copy to avoid modifying the original DataFrame
    return result.copy()


def load_data_split(dataset_to_load, split, data_dir=os.path.join(ROOT_DIR, 'datasets/')):
    if dataset_to_load == 'fetaqa':
        # Load FetaQA directly from the released JSON, bypassing the script-based
        # `datasets` loader (datasets>=3 rejects `fetaqa.py`). This reproduces
        # exactly the structure that datasets/fetaqa.py::_generate_examples emits:
        #   {id, table:{id,header,rows,page_title}, question, answer}
        # FetaQA has a single released file (fetaQA-v1_test.json); all splits map
        # to it (matching datasets/fetaqa.py's SplitGenerators).
        candidate_paths = [
            os.path.join(data_dir, "data", "fetaQA-v1_test.json"),
            "/home/yanmy/HybridRAG/H-STAR/datasets/data/fetaQA-v1_test.json",
        ]
        feta_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if feta_path is None:
            raise FileNotFoundError(
                "fetaQA-v1_test.json not found in: " + ", ".join(candidate_paths))
        dataset = []
        with open(feta_path, encoding="utf-8") as f:
            lines = json.load(f)
        for i, dic in enumerate(lines['data']):
            feta_id = dic['feta_id']
            caption = dic['table_page_title']
            question = dic['question']
            answer = dic["answer"]
            header = dic['table_array'][0]
            rows = dic['table_array'][1:]
            dataset.append({
                "id": feta_id,
                "table": {
                    "id": feta_id,
                    "header": header,
                    "rows": rows,
                    "page_title": caption,
                },
                "question": question,
                "answer": answer,
            })
        return dataset
    if dataset_to_load == 'tab_fact' and split == 'test_small':
        dataset = []
        with open(os.path.join("../utils", "tab_fact", "small_test.jsonl"), "r") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                dic = json.loads(line)
                id = dic['table_id']
                caption = dic['table_caption']
                question = dic['statement']
                answer_text = dic['label']
                header = dic['table_text'][0]
                rows = dic['table_text'][1:]
                
                data = {
                    "id": i,
                    "table": {
                        "id": id,
                        "header": header,
                        "rows": rows,
                        "page_title": caption
                    },
                    "question": question,
                    "answer_text": answer_text
                }
                dataset.append(data)
        return dataset
    else:
        try:
            dataset_split_loaded = load_dataset(
                path=os.path.join(data_dir, "{}.py".format(dataset_to_load)),
                cache_dir=os.path.join(data_dir, "data"))[split]
        except Exception:
            # datasets>=3 removed script-based loading; load the cached arrow directly
            import glob as _glob
            from datasets import Dataset as _DS, concatenate_datasets as _cat
            pat = os.path.join(data_dir, "data", dataset_to_load, "**",
                               "{}-{}.arrow".format(dataset_to_load, split))
            arrows = sorted(_glob.glob(pat, recursive=True))
            if not arrows:
                raise
            dataset_split_loaded = (_cat([_DS.from_file(a) for a in arrows])
                                    if len(arrows) > 1 else _DS.from_file(arrows[0]))

        # unify names of keys
        if dataset_to_load in ['wikitq', 'has_squall', 'missing_squall',
                            'wikitq', 'wikitq_sql_solvable', 'wikitq_sql_unsolvable',
                            'wikitq_sql_unsolvable_but_in_squall',
                            'wikitq_scalability_ori',
                            'wikitq_scalability_100rows',
                            'wikitq_scalability_200rows',
                            'wikitq_scalability_500rows',
                            'wikitq_robustness',
                            'fetaqa'
                            ]:
            pass
        elif dataset_to_load == 'tab_fact':
            new_dataset_split_loaded = []
            for data_item in dataset_split_loaded:
                data_item['question'] = data_item['statement']
                data_item['answer_text'] = data_item['label']
                data_item['table']['page_title'] = data_item['table']['caption']
                new_dataset_split_loaded.append(data_item)
            dataset_split_loaded = new_dataset_split_loaded
        elif dataset_to_load == 'hybridqa':
            new_dataset_split_loaded = []
            for data_item in dataset_split_loaded:
                data_item['table']['page_title'] = data_item['context'].split(' | ')[0]
                new_dataset_split_loaded.append(data_item)
            dataset_split_loaded = new_dataset_split_loaded
        elif dataset_to_load == 'mmqa':
            new_dataset_split_loaded = []
            for data_item in dataset_split_loaded:
                data_item['table']['page_title'] = data_item['table']['title']
                new_dataset_split_loaded.append(data_item)
            dataset_split_loaded = new_dataset_split_loaded
        else:
            raise ValueError(f'{dataset_to_load} dataset is not supported now.')
        return dataset_split_loaded
    
def process_error_analysis_list(error_analysis_list, truncate=True, tau=0.82):
    """
    对一个 list 中的每一个 dict 元素的 'result' 键进行排序和筛选。

    Args:
        error_analysis_list (list): 一个列表，其中每个元素都是一个
                                     包含 'result' 字典的字典。
        truncate (bool, optional): 是否截断结果。默认为 True。
        tau (float, optional): 筛选结果的阈值。默认为 0.82。

    Returns:
        dict: 一个字典，其键是原始列表的索引，值是处理后的结果列表。
    """
    ranked_result = {}

    # 使用 enumerate 遍历列表以获取索引 (index) 和值 (row)
    for index, row in enumerate(error_analysis_list):
        # 检查 'result' 键是否存在
        if 'result' not in row or not isinstance(row['result'], dict):
            ranked_result[index] = [] # 或者可以记录一个错误
            continue

        result_dict = row['result']

        # 1. 按值从大到小排序
        sorted_items = sorted(result_dict.items(), key=lambda item: item[1], reverse=True)
        sorted_keys = [item[0] for item in sorted_items]

        # 2. 如果 truncate=False，直接存储完整排序结果
        if not truncate:
            ranked_result[index] = sorted_keys
            continue # 继续处理列表中的下一个元素

        # 3. truncate 为 True 的情况
        if sorted_keys and sorted_keys[0] == 'Base':
            ranked_result[index] = ['Base']
        else:
            # 移除 'Base' 并取前三位
            other_methods = [k for k in sorted_keys if k != 'Base']
            top_three = other_methods[:3]

            # 按阈值 tau 过滤
            filtered_results = [k for k in top_three if result_dict[k] >= tau]

            # 如果过滤后没有结果，则返回前三位中的第一位
            if not filtered_results:
                if top_three:
                    ranked_result[index] = [top_three[0]]
                else:
                    ranked_result[index] = [] # 如果除了Base没有其他方法
            else:
                # 如果 'Execute_SQL' 在过滤结果中，则将其移到末尾
                if 'Execute_SQL' in filtered_results:
                    filtered_results.remove('Execute_SQL')
                    filtered_results.append('Execute_SQL')
                ranked_result[index] = filtered_results

    return ranked_result