import json
from utils.evaluator import Evaluator
import numpy as np
import pandas as pd
import re
import regex as re
import copy
from utils.normalizer import post_process_sql
from utils.utils import load_data_split
import os
## normalized
from utils.normalizer import convert_df_type
# from utils.optimize_normalizer import convert_df_type
import pandas as pd
# import re
from typing import List

import yaml
from openai import OpenAI
from pathlib import Path
def load_config(path: str = "llm_config.yaml") -> dict:
    """读取配置文件，返回配置字典"""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def call_llm(prompt: str, config_path: str = "../llm_config.yaml") -> str:
    """兼容 OpenAI API 的调用函数"""
    # 读取配置
    config = load_config(config_path)
    model = config.get("model")
    api_key = config.get("api_key")
    base_url = config.get("base_url")

    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 调用接口
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
def print_origin_df(index):
    output_df = pd.DataFrame(eval(outputs[index]['table'])['rows'],columns = eval(outputs[index]['table'])['header'])
    return output_df

def _create_table_prompt(df: pd.DataFrame, title: str):
    """
    Return the CREATE TABLE clause as prompt.
    """
    string = "CREATE TABLE {}(\n".format(title)
    for header in df.columns:
        column_type = 'text'
        try:
            if df[header].dtype == 'int64':
                column_type = 'int'
            elif df[header].dtype == 'float64':
                column_type = 'real'
            elif df[header].dtype == 'datetime64':
                column_type = 'datetime'
        except AttributeError as e:
            raise DuplicateColumnsError(e)

        string += '\t{} {},\n'.format(header, column_type)
    string = string.rstrip(',\n') + ')\n'
    return string

def format_prompt(df: pd.DataFrame, statement: str) -> str:
    """
    将一个 DataFrame 和一个 statement 语句格式化为一个指定的 prompt 字符串。

    这个函数不执行任何逻辑判断，只负责格式化。

    Args:
        df: 输入的 pandas DataFrame。
        statement: 描述数据需求的自然语言语句。

    Returns:
        一个完整格式化的 prompt 字符串。
    """
    # 1. 将 DataFrame 转换为对齐的字符串
    df_string = df.to_string()

    # 2. 获取列名列表并格式化为字符串
    columns_list_str = str(df.columns.tolist())

    # 3. 使用 f-string 组装所有部分到最终的模板中
    prompt = f"""/*
SELECT * FROM w;
{df_string}
*/
columns: {columns_list_str}
statement: {statement}
<output>"""
    
    return prompt
def format_table_prompt(table_title: str, table: pd.DataFrame, statement: str) -> str:
    """
    Build a prompt in the format:

    <input>
    table caption: {table_title}
    /*
    col : c1 | c2 | ...
    row {idx0} : v11 | v12 | ...
    row {idx1} : v21 | v22 | ...
    */
    columns: ['c1', 'c2', ...]
    Q: {statement}
    <output>

    - Uses the DataFrame's existing index values in the "row {idx} :" lines.
    - Converts all cells to strings, strips whitespace, collapses internal newlines, and
      replaces pipe characters in cells to avoid breaking the visual delimiter.
    """
    if table is None:
        raise ValueError("`table` must be a pandas DataFrame, got None.")
    if not isinstance(table, pd.DataFrame):
        # Try to coerce common inputs
        table = pd.DataFrame(table)

    # Prepare a safe stringifier for headers and cells
    def _clean(s) -> str:
        s = "" if pd.isna(s) else str(s)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        s = re.sub(r"\s+", " ", s)
        # Avoid breaking the column/cell delimiter
        s = s.replace("|", "/")
        return s

    cols: List[str] = [ _clean(c) for c in table.columns.tolist() ]
    header_line = "col : " + " | ".join(cols)

    # Ensure values are strings and cleaned
    # (do not reset index; we keep the user's index values)
    row_lines = []
    for idx, row in table.iterrows():
        values = [ _clean(v) for v in row.tolist() ]
        row_lines.append(f"row {idx} : " + " | ".join(values))

    # Compose the block
    lines = []
    lines.append("<input>")
    lines.append(f"table caption: {table_title}")
    lines.append("/*")
    lines.append(header_line)
    lines.extend(row_lines)
    lines.append("*/")
    lines.append(f"columns: {repr(cols)}")
    lines.append(f"Q: {statement}")
    lines.append("<output>")
    return "\n".join(lines)

def build_wikitq_prompt_from_df(dataset,df_input, index, template_path = '../prompts/col_select_sql.txt', processed = False): ## 
    template_text = Path(template_path).read_text(encoding='utf-8')
    if not processed: 
        table_df = convert_df_type(df_input)
    else:
        table_df = df_input
    if template_path.__contains__('sql'):
        table_prompt = _create_table_prompt(table_df, dataset[index]['table']['page_title'])
        table_content_prompt = format_prompt(table_df, dataset[index]['question'])
        return template_text + '\n<input>\n' + table_prompt + '\n' + table_content_prompt
    elif template_path.__contains__('text_reason'):
        table_prompt = format_table_prompt(dataset[index]['table']['page_title'], table_df, dataset[index]['question'])
        return template_text + '\n' + table_prompt
    else:
        table_prompt = format_table_prompt(dataset[index]['table']['page_title'], table_df, dataset[index]['question'])
        return template_text + '\n' + table_prompt

import pandas as pd
import re
from typing import List, Tuple, Any

def filter_dataframe_from_responses(
    response_list: List[str], 
    df: pd.DataFrame,
    add_row_id = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filters a DataFrame based on headers extracted from response texts,
    ignoring case and backslash escape characters.

    Args:
        response_list: A list of model response texts.
        df: The original Pandas DataFrame.
    

    Returns:
        A tuple containing:
        - The filtered subtable (pd.DataFrame).
        - The list of column names in the subtable (List[str]).
    """
    # 1. Prepare DataFrame and create a case-insensitive column map
    df_with_id = df.copy()
    if add_row_id:
        if 'row_id' not in df_with_id.columns:
            df_with_id.insert(0, 'row_id', df_with_id.index)
    
    # Create a map from lowercase column name to its original case
    # e.g., {'air date': 'Air Date', 'title': 'Title'}
    column_map = {col.lower(): col for col in df_with_id.columns}
    df_columns_lower = set(column_map.keys())

    # 2. Define regex patterns
    backtick_pattern = r'`([^`]+)`'
    fcol_pattern = r'f_col\(\[([^\]]+)\]\)'
    
    extracted_headers_lower = set()

    # 3. Iterate through all responses to extract headers
    for response in response_list:
        # Preprocess the string: remove backslashes and convert to lowercase
        processed_response = response.replace('\\', '').lower()

        # Extract headers from `...`
        backtick_matches = re.findall(backtick_pattern, processed_response)
        for header in backtick_matches:
            extracted_headers_lower.add(header.strip())

        # Extract headers from f_col([...])
        fcol_matches = re.findall(fcol_pattern, processed_response)
        for match in fcol_matches:
            headers_in_fcol = match.split(',')
            for header in headers_in_fcol:
                cleaned_header = header.strip().strip("'\"")
                extracted_headers_lower.add(cleaned_header)
    
    # 4. Validate extracted headers against the DataFrame's columns (case-insensitively)
    valid_headers_lower = extracted_headers_lower.intersection(df_columns_lower)

    # 5. Map the valid lowercase headers back to their original case
    final_headers_original_case = {column_map[h] for h in valid_headers_lower}
    
    # 6. Build the final list of columns, ensuring 'row_id' is first
    if add_row_id:
        final_headers = ['row_id'] + sorted([
            h for h in final_headers_original_case if h.lower() != 'row_id'
        ])
    else:
        final_headers = final_headers_original_case

    # 7. Create the subtable using the original-cased column names
    subtable = df_with_id[final_headers]

    return subtable, final_headers

# 3. 调用函数
# filtered_table, final_headers = filter_dataframe_from_responses(response_list, original_df)

# # 4. 打印结果
# print("--- 原始DataFrame ---")
# print(original_df)
# print("\n" + "="*30 + "\n")

# print("--- 提取到的Headers ---")
# # 预期提取到: 'air date', 'title', 'series'
# print(f"最终筛选的列名: {final_headers}")
# print("\n" + "="*30 + "\n")


# print("--- 筛选后的Subtable ---")
# # 预期输出一个包含 'row_id', 'air date', 'series', 'title' 列的DataFrame
# print(filtered_table)

from tqdm import tqdm

# Assuming the Evaluator class is defined elsewhere in your code
# from your_evaluator_module import Evaluator

def evaluate_predictions(dataset_name, df, dataset):
    acc =0
    error_index = []
    format_error_index = []
    # dataset_name = 'wikitq'
    dataset_df_output = df
    outputs = dataset_df_output.to_dict(orient='index')
    # Iterate through each output and calculate the accuracy
    count = 0
    for i in tqdm(outputs.keys()):
        count += 1
        output = outputs[i]
        pred_answer = None
        pred_answer_all = None

        # --- Prediction Parsing Logic ---
        if dataset_name == 'wikitq' or dataset_name == 'niat':
            try:
                # predict column may be a raw string OR a list-repr; handle both
                raw = output['predict']
                if isinstance(raw, str) and raw.strip().startswith('['):
                    try:
                        raw_output = eval(raw)[0]
                    except Exception:
                        raw_output = raw
                else:
                    raw_output = str(raw)
                # Robust final-answer extraction: take the LAST answer marker and bound
                # to that line only (verbose CoT says "the answer is" many times; the old
                # first-occurrence + eval() parser broke on verbose multi-quote outputs).
                marks = list(re.finditer(
                    r'(?:therefore,\s*the answer is|the answer is|final answer)\s*:?\s*',
                    raw_output, re.IGNORECASE))
                if not marks:
                    raise ValueError("Unrecognized format")
                pred_answer_all = raw_output[marks[-1].end():]
                line = pred_answer_all.split('\n')[0]
                quoted = re.findall(r'"([^"]+)"', line)
                if quoted:
                    pred_answer = [x.strip() for x in quoted]
                else:
                    cleaned = line.strip().strip('.').strip()
                    pred_answer = [cleaned] if cleaned else ['']

            except Exception as e:
                # If parsing fails, log it as a format error
                format_error_index.append(i)
                pred_answer = [''] # Assign an empty prediction

        elif dataset_name == 'tab_fact':
            # Robust: predict may be raw string or list-repr; model emits the verdict as
            # `the answer is: "yes"/"no"` (or true/false/supported). Old parser eval()'d the
            # raw string (crash) and split on 'statement is: ' (absent) -> broken.
            raw = output['predict']
            if isinstance(raw, str) and raw.strip().startswith('['):
                try:
                    raw_output = eval(raw)[0]
                except Exception:
                    raw_output = raw
            else:
                raw_output = str(raw)
            marks = list(re.finditer(
                r'(?:therefore,\s*the answer is|the answer is|final answer|statement is)\s*:?\s*',
                raw_output, re.IGNORECASE))
            seg = raw_output[marks[-1].end():].split('\n')[0] if marks else raw_output
            low = seg.lower()
            quoted = re.findall(r'"([^"]+)"', seg)
            ans = quoted[0].lower() if quoted else low
            if re.search(r'\b(no|false|refut|incorrect|not\s+support)\b', ans):
                pred_answer = [0]
            elif re.search(r'\b(yes|true|support|correct)\b', ans):
                pred_answer = [1]
            else:
                format_error_index.append(i)
                pred_answer = [1 if ('true' in low or 'yes' in low or 'support' in low) else 0]

            # gold_answer = result_df[str(i)]['ori_data_item']['answer_text']
        gold_answer = dataset[i]['answer_text']
        # Score is either 1 or 0
        score = Evaluator().evaluate(
        pred_answer,
        gold_answer,
        dataset=dataset_name,
        question=output['question']
        )
        if dataset_name == 'wikitq' or dataset_name == 'niat':
            if score == False and isinstance(pred_answer_all, str):
                score = Evaluator().evaluate(
                    pred_answer_all.split(','),
                    gold_answer,
                    dataset=dataset_name,
                    question=output.get('question', '') # Safely get question if available
                )
        acc += score
        if score != 1:
            error_index.append(i)
            # if score!=1:
            #     print(f'The prediction is {pred_answer}, while the ground truth is {gold_answer} for sample {i}.')
        # except:
        #     print(i)
        #     pass
    final_accuracy = 100 * acc / count if count > 0 else 0
    print(f"Correct Samples: {acc}; Total Samples: {count}")
    print(f"Accuracy: {100*acc/count:.2f}")
    return final_accuracy, error_index, format_error_index

# --- Example Usage ---
# Assuming you have:
# 1. `output_df`: A pandas DataFrame with model predictions.
# 2. `my_dataset`: Your loaded dataset (e.g., from Hugging Face datasets).
# 3. `Evaluator`: The evaluation class is defined.

# accuracy, prediction_errors, formatting_errors = evaluate_predictions(
#     dataset_name='wikitq', 
#     df=output_df, 
#     dataset=my_dataset
# )

# print(f"\nPrediction errors occurred at indices: {prediction_errors}")
# print(f"Format errors occurred at indices: {formatting_errors}")
import re
import pandas as pd

def fix_sql_query(response_text: str, table_df: pd.DataFrame, table_title: str) -> str:
    """
    Extracts the last SQL query from text, cleans it, and standardizes it.

    Fixes include:
    1. Replaces the explicit table_title with 'w'. Handles both the original title 
       ("Luís Sá Silva") and slugified versions ("Luís_Sá_Silva").
    2. Wraps multi-word column names with backticks `` ` ``.
    3. Handles escapes, newlines, and ensures a trailing semicolon.
    (Version 2: Correctly handles semicolons in subqueries)
    """
    # --- 1. Get lowercase column names from the DataFrame ---
    df = table_df.copy()
    df.columns = [str(col).lower().strip() for col in df.columns]
    table_columns = df.columns.tolist()

    # --- 2. Extract the last SQL query from the response text ---
    # This regex is improved to better capture the final intended SQL block
    code_blocks = re.findall(r'```(?:sql)?\s*(.*?)\s*```', response_text, re.DOTALL)
    raw_sql = ""
    if code_blocks:
        raw_sql = code_blocks[-1]
    else:
        # Prioritize finding a line that explicitly starts with "SQL:"
        sql_matches = re.findall(r'(?im)^\s*SQL:\s*(.*)', response_text)
        if sql_matches:
            raw_sql = sql_matches[-1]
        else:
            # Fallback to the previous method if "SQL:" is not found
            sql_matches = re.findall(r'(?i)(?:SQL|Response \d+):\s*(.*)', response_text, re.DOTALL)
            if sql_matches:
                raw_sql = sql_matches[-1]
            else:
                last_select_pos = response_text.lower().rfind('select ')
                if last_select_pos != -1:
                    raw_sql = response_text[last_select_pos:]
    if not raw_sql:
        return ""

    # --- 3. Clean and Standardize the SQL string ---
    cleaned_sql = raw_sql.replace('\n', ' ').strip()
    cleaned_sql = cleaned_sql.replace("\\'", "''")
    
    # --- 4. Replace the table title (original or slugified) with 'w' ---
    slugified_title = re.sub(r'[\s\W]+', '_', table_title)
    title_pattern = re.compile(
        f"""(?:'|\"|`)?({re.escape(table_title)}|{re.escape(slugified_title)})(?:'|\"|`)?""", 
        re.IGNORECASE
    )
    sql_with_w = title_pattern.sub('w', cleaned_sql)

    # --- 5. Wrap multi-word column names ---
    fixed_sql = sql_with_w
    for col_name in table_columns:
        if ' ' in col_name:
            pattern = re.compile(r'\b' + re.escape(col_name) + r'\b', re.IGNORECASE)
            fixed_sql = pattern.sub(f'`{col_name}`', fixed_sql)

    # --- 6. [MODIFIED] Finalize the query robustly ---
    # Find the last semicolon and take everything before it. This prevents
    # premature splitting on semicolons within subqueries.
    last_semicolon_pos = fixed_sql.rfind(';')
    if last_semicolon_pos != -1:
        # Trim the string to just after the last semicolon
        final_sql = fixed_sql[:last_semicolon_pos + 1].strip()
    else:
        # If no semicolon exists, add one
        final_sql = fixed_sql.strip() + ';'
    
    return final_sql
def add_row_id_to_select(sql_query: str) -> str:
    """
    Modifies a SQL query string to include 'row_id' in the SELECT clause.

    This function is case-insensitive and handles several edge cases:
    - It will not modify a query that already selects 'row_id'.
    - It will not modify a 'SELECT DISTINCT' query to preserve its unique results.
    - It correctly modifies 'SELECT *' to 'SELECT row_id, *'.

    Args:
        sql_query: The original SQL query string.

    Returns:
        The modified SQL query with 'row_id' added to the selection.
    """
    # Normalize the query for checks (case-insensitive, no leading/trailing whitespace)
    normalized_query = sql_query.strip().lower()

    # If it's not a SELECT query, or uses DISTINCT, or already has row_id, return it unchanged.
    if not normalized_query.startswith('select '):
        return sql_query
    if normalized_query.startswith('select distinct'):
        return sql_query
    # Check for 'row_id' only in the initial SELECT part of the query
    if 'row_id' in normalized_query.split('from')[0]:
        return sql_query

    # Use regex for a robust, case-insensitive replacement of the "SELECT" keyword
    # This replaces the first occurrence of "SELECT " with "SELECT row_id, "
    modified_query = re.sub(r'^\s*SELECT\s+', 'SELECT row_id, ', sql_query.strip(), count=1, flags=re.IGNORECASE)
    
    return modified_query
def match_subtables(large_table: pd.DataFrame, sub_tables: list) -> pd.DataFrame:
    """
    在 large_table 中查找所有出现在 sub_tables 中的行（忽略大小写，模糊匹配）。

    参数:
    - large_table: pd.DataFrame
    - sub_tables: list[pd.DataFrame]

    返回:
    - 匹配到的 large_table 子集 DataFrame
    """
    if not isinstance(large_table, pd.DataFrame):
        raise ValueError("large_table 必须是 DataFrame")
    if not isinstance(sub_tables, list):
        raise ValueError("sub_tables 必须是 DataFrame 列表")

    # 确保所有列都是字符串，便于比较
    large_str = large_table.astype(str).apply(lambda col: col.str.lower())

    # 存储所有匹配到的行索引
    matched_idx = set()

    for sub in sub_tables:
        sub_str = sub.astype(str).apply(lambda col: col.str.lower())

        # 遍历 sub_table 的每一行
        for _, sub_row in sub_str.iterrows():
            cond = pd.Series([True] * len(large_str))
            for col in sub_row.index:
                if col in large_str.columns:
                    # 使用正则模糊匹配（忽略大小写）
                    pattern = re.escape(str(sub_row[col]))
                    cond &= large_str[col].str.contains(pattern, case=False, na=False)
            matched_idx.update(large_str[cond].index.tolist())

    return large_table.loc[sorted(matched_idx)].reset_index(drop=True)
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

def _evaluate_row(row, dataset_name, dataset, evaluator):
    """
    辅助函数，用于处理和评估 DataFrame 的单行数据。

    Args:
        row (pd.Series): DataFrame 的单行。
        dataset_name (str): 数据集名称 ('wikitq' 或 'tab_fact')。
        dataset (dict or list): 包含标准答案的原始数据集。
        evaluator (Evaluator): Evaluator 类的实例。

    Returns:
        pd.Series: 包含分数（1或0）和格式错误标志的 Series。
    """
    pred_answer = None
    is_format_error = False
    
    # --- 1. 预测解析逻辑 ---
    if dataset_name == 'wikitq':
        pred_answer_all_str = None
        try:
            # 尝试解析多种可能的输出格式
            raw_output = eval(row['predict'])[0]
            if 'the answer is: ' in raw_output:
                pred_answer_all_str = raw_output.split('the answer is: ')[1]
            elif 'Final Answer: ' in raw_output:
                pred_answer_all_str = raw_output.split('Final Answer: ')[1]
            else:
                # 如果没有特定关键词，则假定整个字符串为答案
                pred_answer_all_str = raw_output

            # 提取主要答案，通常在引号中
            pred_answer = pred_answer_all_str.split('"')[1:2]
            if not pred_answer or pred_answer == ['']: # 处理分割失败或结果为空的情况
                pred_answer = [pred_answer_all_str]

        except Exception as e:
            # 解析失败则记录为格式错误
            is_format_error = True
            pred_answer = [''] # 分配一个默认的空预测

    elif dataset_name == 'tab_fact':
        # 对 TabFact 数据集，不区分大小写地检查 'true' 或 'false'
        if isinstance(row['predict'], str) and 'true' in row['predict'].lower():
            pred_answer = [1]
        else:
            pred_answer = [0]
            
    # --- 2. 评估 ---
    gold_answer = dataset[row.name]['answer_text']
    
    # 计算分数 (正确为1, 错误为0)
    score = evaluator.evaluate(
        pred_answer,
        gold_answer,
        dataset=dataset_name,
        question=row.get('question', '')
    )
    
    # 为 WikiTQ 设计的备用评估逻辑，尝试用逗号分割答案
    if not score and dataset_name == 'wikitq' and pred_answer_all_str is not None:
        score = evaluator.evaluate(
            [s.strip() for s in pred_answer_all_str.split(',')],
            gold_answer,
            dataset=dataset_name,
            question=row.get('question', '')
        )

    return pd.Series([int(score), is_format_error], index=['score', 'is_format_error'])

# def evaluate_predictions(dataset_name, df, dataset, use_pandarallel=False, workers=16):
#     """
#     使用 pandas.apply 高效评估模型预测结果，并可选并行处理。

#     Args:
#         dataset_name (str): 数据集名称 ('wikitq' 或 'tab_fact')。
#         df (pd.DataFrame): 包含模型输出的 DataFrame，必须有 'predict' 列。
#         dataset (dict or list): 包含标准答案的原始数据集。
#         use_pandarallel (bool): 若为 True，则使用 pandarallel 并行执行。
#         workers (int): pandarallel 使用的 worker 数量。

#     Returns:
#         tuple: 一个元组，包含:
#             - float: 最终的准确率（百分比）。
#             - list: 预测错误的样本索引列表。
#             - list: 预测格式错误的样本索引列表。
#     """
#     if df.empty:
#         print("输入 DataFrame 为空，返回准确率为 0。")
#         return 0.0, [], []
        
#     print(f"开始评估 '{dataset_name}' 数据集...")
    
#     # 仅实例化一次 Evaluator
#     evaluator = Evaluator()
    
#     # --- 选择执行方式 ---
#     if use_pandarallel:
#         print(f"使用 pandarallel 并行处理，worker 数量: {workers}。")
#         pandarallel.initialize(nb_workers=workers, progress_bar=True)
#         apply_method = df.parallel_apply
#     else:
#         print("使用标准 pandas apply 并显示进度条。")
#         tqdm.pandas(desc=f"正在评估 {dataset_name}")
#         apply_method = df.progress_apply

#     # --- 对每一行应用评估函数 ---
#     results_df = apply_method(
#         lambda row: _evaluate_row(row, dataset_name, dataset, evaluator),
#         axis=1
#     )
    
#     # --- 计算最终结果 ---
#     acc = results_df['score'].sum()
#     count = len(df)
    
#     final_accuracy = 100 * acc / count if count > 0 else 0
    
#     # 获取错误样本的索引
#     error_index = results_df[results_df['score'] == 0].index.tolist()
#     format_error_index = results_df[results_df['is_format_error']].index.tolist()
    
#     print("\n--- 评估完成 ---")
#     print(f"正确样本数: {acc}; 总样本数: {count}")
#     print(f"准确率: {final_accuracy:.2f}%")
    
#     return final_accuracy, error_index, format_error_index

import pandas as pd
from pathlib import Path
from typing import Union, List, Dict

# Assuming 'Dataset' is a type hint for a list of dictionaries, 
# similar to what Hugging Face datasets produce.
Dataset = List[Dict]



def build_single_prompt_fetaqa(
    dataframe: pd.DataFrame, 
    dataset: Dataset, 
    index: int, 
    few_shot_path: str
) -> str:
    """
    Builds a single prompt for a given item using a pandas DataFrame for the table.
    This version is for the FeTaQA format.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the table data.
        dataset (Dataset): The original dataset (list of dicts) to get metadata.
        index (int): The index of the item to process from the dataset.
        few_shot_path (str): Path to the few-shot examples file.

    Returns:
        str: The fully formatted prompt string for the specified item.
    """
    # Read the few-shot prompt template
    few_shot = Path(few_shot_path).read_text(encoding="utf-8").strip()

    # Get the specific item from the dataset for metadata
    item = dataset[index]
    statement = item["question"]
    table_caption = item["table"]['page_title']
    sub_title = item["table"]['subsection_title']

    # Format the table from the DataFrame
    headers = 'col : ' + " | ".join(dataframe.columns)
    # Iterate through DataFrame rows and join values, adding row index
    rows = [
        f'row {i} : ' + " | ".join(map(str, row))
        for i, row in enumerate(dataframe.itertuples(index=False))
    ]
    table_text = headers + "\n" + "\n".join(rows)

    # Assemble the final prompt
    prompt = (
        few_shot
        + f"\n\n<input>\ntable caption: {table_caption}\nsub_title: {sub_title}\n/*\n{table_text}\n*/\nquestion: {statement}\n<output>\n"
    )

    return prompt

def build_tab_fact_prompt_from_df(
    dataframe: pd.DataFrame, 
    dataset: Dataset, 
    index: int, 
    template_path: str
) -> str:
    """
    根据给定的数据集、DataFrame、索引和模板路径，构建一个特定格式的 prompt。

    此函数结合了以下特点：
    - 从 template_path 加载 few-shot/模板内容。
    - 表格从外部 DataFrame 传入，并格式化为 "col : ... / row 0 : ..." 样式。
    - 表格部分被包裹在 /* ... */ 注释块中。
    - 问题/论述以 "statement:" 为前缀。

    Args:
        dataset (Dataset): 原始数据集 (list of dicts)，用于获取 statement。
        dataframe (pd.DataFrame): 包含当前条目表格数据的 DataFrame。
        index (int): 当前条目在原始数据集中的索引。
        template_path (str): few-shot 示例或模板文件的路径。

    Returns:
        str: 整合了模板和特定格式化内容的最终 prompt 字符串。
    """
    # 1. 读取模板/few-shot 部分
    template = Path(template_path).read_text(encoding="utf-8").strip()

    # 2. 从数据集中获取 statement
    item = dataset[index]
    statement = item["question"]

    # 3. 格式化 DataFrame 为指定的文本样式
    headers = 'col : ' + " | ".join(dataframe.columns)
    rows = [
        f'row {i} : ' + " | ".join(map(str, row))
        for i, row in enumerate(dataframe.itertuples(index=False))
    ]
    table_text = headers + "\n" + "\n".join(rows)

    # 4. 组装最终的 prompt，结构为：模板 + 您指定的新格式
    prompt = (
        template
        + "\n\n"
        + f"<input>\n/*\n{table_text}\n*/\n"
        + f"statement: {statement}\n<output>"
    )

    return prompt
