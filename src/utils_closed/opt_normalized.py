# fast_df_utils.py
# -*- coding: utf-8 -*-
"""
Drop-in faster replacement focused on speeding up `convert_df_type`.
Key changes:
- Vectorized operations (replace/str ops) instead of per-cell loops
- One-pass column typing using pandas.to_numeric / pandas.to_datetime
- Deduplicate expensive `str_normalize` calls via LRU cache + per-column unique mapping
- O(1) duplicate header resolution with counters
"""

from typing import List, Dict
import re
import unicodedata
from functools import lru_cache
from collections import Counter

import numpy as np
import pandas as pd

import recognizers_suite
from recognizers_suite import Culture

from fuzzywuzzy import fuzz

from utils.sql.extraction_from_sql import *
from utils.sql.all_keywords import ALL_KEY_WORDS

culture = Culture.English


# =========================
# Normalization utilities
# =========================

def str_normalize(user_input, recognition_types=None):
    """A string normalizer which recognize and normalize value based on recognizers_suite"""
    user_input = str(user_input)
    user_input = user_input.replace("\\n", "; ")

    def replace_by_idx_pairs(orig_str, strs_to_replace, idx_pairs):
        assert len(strs_to_replace) == len(idx_pairs)
        last_end = 0
        to_concat = []
        for idx_pair, str_to_replace in zip(idx_pairs, strs_to_replace):
            to_concat.append(orig_str[last_end:idx_pair[0]])
            to_concat.append(str_to_replace)
            last_end = idx_pair[1]
        to_concat.append(orig_str[last_end:])
        return ''.join(to_concat)

    if recognition_types is None:
        recognition_types = [
            "datetime",
            "number",
            # "ordinal",
            # "percentage",
            # "age",
            # "currency",
            # "dimension",
            # "temperature",
        ]

    for recognition_type in recognition_types:
        if re.match(r"\d+/\d+", user_input):
            # avoid calculating str as 1991/92
            continue
        recognized_list = getattr(recognizers_suite, f"recognize_{recognition_type}")(user_input, culture)
        strs_to_replace, idx_pairs = [], []
        for recognized in recognized_list:
            if recognition_type != 'datetime':
                recognized_value = recognized.resolution['value']
                if str(recognized_value).startswith("P"):  # datetime period
                    continue
                else:
                    strs_to_replace.append(recognized_value)
                    idx_pairs.append((recognized.start, recognized.end + 1))
            else:
                if recognized.resolution:  # sometimes None
                    if len(recognized.resolution['values']) == 1:
                        # Use TIMEX as normalization
                        strs_to_replace.append(recognized.resolution['values'][0]['timex'])
                        idx_pairs.append((recognized.start, recognized.end + 1))

        if strs_to_replace:
            user_input = replace_by_idx_pairs(user_input, strs_to_replace, idx_pairs)

    if re.match(r"(.*)-(.*)-(.*) 00:00:00", user_input):
        # '2008-04-13 00:00:00' -> '2008-04-13'
        user_input = user_input[: -len("00:00:00") - 1]
    return user_input


@lru_cache(maxsize=100_000)
def _str_normalize_cached(x: str) -> str:
    """LRU-cached wrapper around str_normalize (massive speed-up on repeated values)."""
    return str_normalize(x)


def _unique_map_normalize(series: pd.Series) -> pd.Series:
    """
    Normalize a string series by computing normalization ONLY on unique values,
    then mapping back. Avoids calling recognizers repeatedly.
    """
    # Ensure string dtype
    s = series.astype(str)
    uniques = pd.unique(s)
    # Build mapping with cache
    mapping = {u: _str_normalize_cached(u) for u in uniques}
    return s.map(mapping)


# =========================
# Header utilities
# =========================

def _make_unique(names: List[str], sep: str = "_") -> List[str]:
    """
    Make a list of column names unique by appending suffixes.
    Fast O(n) implementation.
    """
    counter = Counter()
    out = []
    for name in names:
        base = name if name != "" else "FilledColumnName"
        if counter[base] == 0:
            out.append(base)
        else:
            out.append(f"{base}{sep}{counter[base]+1}")
        counter[base] += 1
    return out


def _lower_and_make_unique(names: List[str], sep: str = "-") -> List[str]:
    counter = Counter()
    out = []
    for name in names:
        base = str(name).lower()
        if counter[base] == 0:
            out.append(base)
        else:
            out.append(f"{base}{sep}{counter[base]+1}")
        counter[base] += 1
    return out


# =========================
# Main API
# =========================

def prepare_df_for_neuraldb_from_table(table: Dict, add_row_id=True, normalize=True, lower_case=True):
    header, rows = table['header'], table['rows']
    if add_row_id and 'row_id' not in header:
        header = ["row_id"] + header
        rows = [["{}".format(i)] + row for i, row in enumerate(rows)]
    if normalize:
        df = convert_df_type(pd.DataFrame(data=rows, columns=header), lower_case=lower_case)
    else:
        df = pd.DataFrame(data=rows, columns=header)
    return df


def convert_df_type(df: pd.DataFrame, lower_case: bool = True) -> pd.DataFrame:
    """
    Faster converter of DataFrame data type from string to int/float/datetime.

    Pipeline (vectorized):
      1) Fix empty/duplicate headers
      2) Replace null-like tokens globally (['', '-', '/'] -> 'None')
      3) Normalize cells (cached, unique-map)
      4) Strip uniform time suffix if applies to entire column
      5) Lowercase headers & (optionally) cells (vectorized)
      6) Type inference per column using pandas.to_numeric / to_datetime (all-or-nothing)
         - Int -> pandas nullable Int64 (preserves missing)
         - Float -> float64
         - Datetime -> datetime64[ns]
    """
    if df.empty:
        return df

    # 1) Fix empty/duplicate headers (single pass)
    df = df.copy()
    df.columns = _make_unique([str(c) for c in df.columns])

    # 2) Replace null-like tokens for the whole DF (vectorized)
    # Keep behavior consistent with original: null tokens -> str(None)
    null_tokens = ['', '-', '/']
    df = df.replace(null_tokens, str(None))

    # 3) Normalize cell values (heavy) -> do unique mapping per column with cache
    # Only for object columns
    obj_cols = df.columns[df.dtypes == object]
    for col in obj_cols:
        df[col] = _unique_map_normalize(df[col])

    # 4) Strip uniform time suffix if *every* value ends with the same suffix
    # Use vectorized checks
    suffixes = ("-01-01 00:00:00", "-01 00:00:00", " 00:00:00")
    for col in obj_cols:
        s = df[col].astype(str)
        for suf in suffixes:
            if s.str.endswith(suf).all():
                df[col] = s.str.slice(0, -len(suf))
                break  # At most one suffix applies

    # 5) Lowercase headers & cells
    if lower_case:
        df.columns = _lower_and_make_unique(list(df.columns))
        # Lowercase only object columns
        for col in obj_cols:
            # Note: safe even if column converted earlier (still object here)
            df[col] = df[col].astype(str).str.lower().str.strip()

    # 6) Type inference (numeric first, then datetime). "All-or-nothing" to match original.
    def _coerce_numeric(series: pd.Series):
        s = series.astype(str)
        # Treat 'none'/'nan' (case-insensitive) as missing
        missing_mask = s.str.lower().isin(['none', 'nan', ''])
        num = pd.to_numeric(s.mask(missing_mask, np.nan), errors='coerce')
        # All non-missing must be numeric
        if num.notna().sum() == (~missing_mask).sum():
            # Int or float?
            # For ints: all fractional parts are zero
            if (num.dropna() % 1 == 0).all():
                # Use pandas nullable Int64 to preserve missing values
                try:
                    return num.astype('Int64'), True
                except Exception:
                    # Fallback to float if corner cast fails
                    return num.astype(float), True
            else:
                return num.astype(float), True
        return series, False

    def _coerce_datetime(series: pd.Series):
        s = series.astype(str)
        missing_mask = s.str.lower().isin(['none', 'nan', ''])
        dt = pd.to_datetime(s.mask(missing_mask, pd.NaT), errors='coerce', infer_datetime_format=True, utc=False)
        if dt.notna().sum() == (~missing_mask).sum():
            return dt, True
        return series, False

    for col in df.columns:
        # Only attempt on object-like columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            # numeric?
            new_col, ok = _coerce_numeric(df[col])
            if ok:
                df[col] = new_col
                continue
            # datetime?
            new_col, ok = _coerce_datetime(df[col])
            if ok:
                df[col] = new_col
                continue
        # else leave as-is

    return df


# =========================
# Extra helpers (unchanged)
# =========================

def normalize(x):
    """ Normalize string. """
    # Copied from WikiTableQuestions dataset official evaluator.
    if x is None:
        return None
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x) if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub("[‘’´`]", "'", x)
    x = re.sub("[“”]", "\"", x)
    x = re.sub("[‐-‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub("((?<!^)$begin:math:display$[^$end:math:display$]*\]|$begin:math:display$\\d+$end:math:display$|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub("(?<!^)( $begin:math:text$[^)]*$end:math:text$)*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub('^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


def post_process_sql(sql_str, df, table_title=None, process_program_with_fuzzy_match_on_db=True, verbose=False):
    """Post process SQL: including basic fix and further fuzzy match on cell and SQL to process"""

    def basic_fix(sql_str, all_headers, table_title=None):
        def finditer(sub_str: str, mother_str: str):
            result = []
            start_index = 0
            while True:
                start_index = mother_str.find(sub_str, start_index, -1)
                if start_index == -1:
                    break
                end_idx = start_index + len(sub_str)
                result.append((start_index, end_idx))
                start_index = end_idx
            return result

        if table_title:
            sql_str = sql_str.replace("FROM " + table_title, "FROM w")
            sql_str = sql_str.replace("FROM " + '`' + table_title + '`', "FROM w")
            sql_str = sql_str.replace("FROM " + table_title.lower(), "FROM w")
            sql_str = sql_str.replace("FROM " + '`' + table_title.lower() + '`', "FROM w")

        # Remove the null header.
        while '' in all_headers:
            all_headers.remove('')

        # Normalize header newlines
        sql_str = sql_str.replace("\\n", "\n")
        sql_str = sql_str.replace("\n", "\\n")

        # Prepare mask array
        have_matched = [0 for _ in range(len(sql_str))]

        # match quotation
        idx_s_single_quotation = [_ for _ in range(1, len(sql_str)) if sql_str[_] in ["\'"] and sql_str[_ - 1] not in ["\'"]]
        idx_s_double_quotation = [_ for _ in range(1, len(sql_str)) if sql_str[_] in ["\""] and sql_str[_ - 1] not in ["\""]]
        for idx_s in [idx_s_single_quotation, idx_s_double_quotation]:
            if len(idx_s) % 2 == 0:
                for idx in range(int(len(idx_s) / 2)):
                    start_idx = idx_s[idx * 2]
                    end_idx = idx_s[idx * 2 + 1]
                    have_matched[start_idx: end_idx] = [2 for _ in range(end_idx - start_idx)]

        # Add backticks around multi-word headers not inside quotes/keywords
        all_headers.sort(key=lambda x: len(x), reverse=True)

        def _safe_slice(arr, a, b):
            return arr[a:b] if 0 <= a < len(arr) and 0 <= b <= len(arr) else []

        for header in all_headers:
            if (header in sql_str) and (header not in ALL_KEY_WORDS) and (' ' in header):
                for start_idx, end_idx in finditer(header, sql_str):
                    inside_quote = any(v != 0 for v in _safe_slice(have_matched, start_idx, end_idx))
                    left_tick = (start_idx - 1 >= 0 and sql_str[start_idx - 1] == "`")
                    right_tick = (end_idx < len(sql_str) and sql_str[end_idx] == "`")
                    if (not inside_quote) and (not left_tick) and (not right_tick):
                        for i in range(start_idx, end_idx):
                            have_matched[i] = 1

        # re-compose sql from the matched idx.
        start_have_matched = [0] + have_matched
        end_have_matched = have_matched + [0]
        start_idx_s = [idx - 1 for idx in range(1, len(start_have_matched)) if start_have_matched[idx - 1] == 0 and start_have_matched[idx] == 1]
        end_idx_s = [idx for idx in range(len(end_have_matched) - 1) if end_have_matched[idx] == 1 and end_have_matched[idx + 1] == 0]
        assert len(start_idx_s) == len(end_idx_s)
        spans = []
        current_idx = 0
        for start_idx, end_idx in zip(start_idx_s, end_idx_s):
            spans.append(sql_str[current_idx:start_idx])
            spans.append(sql_str[start_idx:end_idx + 1])
            current_idx = end_idx + 1
        spans.append(sql_str[current_idx:])
        sql_str = '`'.join(spans)
        return sql_str

    def fuzzy_match_process(sql_str, df, verbose=False):
        """
        Post-process SQL by fuzzy matching value with table contents.
        """

        def _get_matched_cells(value_str, df, fuzz_threshold=70):
            matched_cells = []
            for _, row in df.iterrows():
                for cell in row:
                    cell = str(cell)
                    fuzz_score = fuzz.ratio(value_str, cell)
                    if fuzz_score == 100:
                        return [(cell, fuzz_score)]
                    if fuzz_score >= fuzz_threshold:
                        matched_cells.append((cell, fuzz_score))
            matched_cells = sorted(matched_cells, key=lambda x: x[1], reverse=True)
            return matched_cells

        def _check_valid_fuzzy_match(value_str, matched_cell):
            number_pattern = r"[+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
            numbers_in_value = re.findall(number_pattern, value_str)
            numbers_in_matched_cell = re.findall(number_pattern, matched_cell)
            try:
                numbers_in_value = [float(num.replace(',', '')) for num in numbers_in_value]
            except Exception:
                print(f"Can't convert number string {numbers_in_value} into float in _check_valid_fuzzy_match().")
            try:
                numbers_in_matched_cell = [float(num.replace(',', '')) for num in numbers_in_matched_cell]
            except Exception:
                print(f"Can't convert number string {numbers_in_matched_cell} into float in _check_valid_fuzzy_match().")
            return set(numbers_in_value).issubset(numbers_in_matched_cell) or set(numbers_in_matched_cell).issubset(numbers_in_value)

        # Drop trailing '\n```'
        sql_str = sql_str.rstrip('```').rstrip('\n')

        # Replace QA module with placeholder
        qa_pattern = r"QA\(.+?;.*?`.+?`.*?\)"
        qas = re.findall(qa_pattern, sql_str)
        for idx, qa in enumerate(qas):
            sql_str = sql_str.replace(qa, f"placeholder{idx}")

        # Parse and replace SQL value with table contents
        sql_tokens = tokenize(sql_str)
        sql_template_tokens = extract_partial_template_from_sql(sql_str)
        # Fix 'between' keyword bug in parsing templates
        fixed_sql_template_tokens = []
        sql_tok_bias = 0
        for idx, sql_templ_tok in enumerate(sql_template_tokens):
            sql_tok = sql_tokens[idx + sql_tok_bias]
            if sql_tok == 'between' and sql_templ_tok == '[WHERE_OP]':
                fixed_sql_template_tokens.extend(['[WHERE_OP]', '[VALUE]', 'and'])
                sql_tok_bias += 2
            else:
                fixed_sql_template_tokens.append(sql_templ_tok)
        sql_template_tokens = fixed_sql_template_tokens
        for idx, tok in enumerate(sql_tokens):
            if tok in ALL_KEY_WORDS:
                sql_tokens[idx] = tok.upper()

        if verbose:
            print(sql_tokens)
            print(sql_template_tokens)

        assert len(sql_tokens) == len(sql_template_tokens)
        value_indices = [idx for idx in range(len(sql_template_tokens)) if sql_template_tokens[idx] == '[VALUE]']
        for value_idx in value_indices:
            # Skip the value if the where condition column is QA module
            if value_idx >= 2 and sql_tokens[value_idx - 2].startswith('placeholder'):
                continue
            value_str = sql_tokens[value_idx]
            # Drop quotes for fuzzy match
            is_string = False
            if value_str and value_str[0] == "\"" and value_str[-1] == "\"":
                value_str = value_str[1:-1]
                is_string = True
            # Already fuzzy
            if value_str and (value_str[0] == '%' or value_str[-1] == '%'):
                continue
            value_str = value_str.lower()

            matched_cells = _get_matched_cells(value_str, df)
            if verbose:
                print(matched_cells)

            new_value_str = value_str
            if matched_cells:
                for matched_cell, fuzz_score in matched_cells:
                    if _check_valid_fuzzy_match(value_str, matched_cell):
                        new_value_str = matched_cell
                        if verbose and new_value_str != value_str:
                            print("\tfuzzy match replacing!", value_str, '->', matched_cell, f'fuzz_score:{fuzz_score}')
                        break
            if is_string:
                new_value_str = f"\"{new_value_str}\""
            sql_tokens[value_idx] = new_value_str

        # Compose new sql string
        new_sql_str = ' '.join(sql_tokens)
        sql_columns = re.findall(r'`\s(.*?)\s`', new_sql_str)
        for sql_col in sql_columns:
            matched_columns = []
            for col in df.columns:
                score = fuzz.ratio(sql_col.lower(), col)
                if score == 100:
                    matched_columns = [(col, score)]
                    break
                if score >= 80:
                    matched_columns.append((col, score))
            matched_columns = sorted(matched_columns, key=lambda x: x[1], reverse=True)
            if matched_columns:
                matched_col = matched_columns[0][0]
                new_sql_str = new_sql_str.replace(f"` {sql_col} `", f"`{matched_col}`")
            else:
                new_sql_str = new_sql_str.replace(f"` {sql_col} `", f"`{sql_col}`")

        # Restore QA modules
        for idx, qa in enumerate(qas):
            new_sql_str = new_sql_str.replace(f"placeholder{idx}", qa)

        # Fix '<>' spacing
        new_sql_str = new_sql_str.replace('< >', '<>')
        return new_sql_str

    sql_str = basic_fix(sql_str, list(df.columns), table_title)
    if process_program_with_fuzzy_match_on_db:
        try:
            sql_str = fuzzy_match_process(sql_str, df, verbose)
        except Exception:
            pass
    return sql_str