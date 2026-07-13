# database_executor.py

import copy
import os
import sqlite3
import records
import sqlalchemy
import pandas as pd
from typing import Dict, List, Any, Union
import uuid
from tqdm.notebook import tqdm

# Helper functions can remain the same
# ...

class NeuralDB(object):
    """
    A database class that holds multiple tables in a single SQLite file.
    The execute_query method is now enhanced to intelligently include 'row_id'.
    """
    def __init__(self, tables: Union[List[pd.DataFrame], Dict[int, pd.DataFrame]], table_titles: List[str] = None):
        if not tables:
            raise ValueError("Cannot initialize NeuralDB with no tables.")

        table_iterator = None
        if isinstance(tables, list):
            table_iterator = tables
        elif isinstance(tables, dict):
            table_iterator = tables.values()
        else:
            raise TypeError(f"Expected 'tables' to be a list or dict of DataFrames, but got {type(tables)}")
        
        self.tables_df = {}
        for i, tbl in enumerate(table_iterator):
            if not isinstance(tbl, pd.DataFrame):
                raise TypeError(f"All items must be pandas DataFrames, but found {type(tbl)} at index {i}")
            self.tables_df[i] = tbl.copy()
        
        self.table_titles = {i: title for i, title in enumerate(table_titles)} if table_titles else {}
        self.table_map = {i: f"w{i}" for i in range(len(self.tables_df))}

        self.tmp_path = "tmp"
        os.makedirs(self.tmp_path, exist_ok=True)
        self.db_path = os.path.join(self.tmp_path, f'{uuid.uuid4()}.db')
        self.sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)

        print("Initializing NeuralDB...")
        for table_id, df in tqdm(self.tables_df.items(), desc="Creating tables in DB"):
            table_name = self.table_map[table_id]
            # This part is crucial: ensure the row_id column exists in the underlying table
            if 'row_id' not in df.columns:
                df_with_id = df.copy()
                df_with_id.insert(0, 'row_id', range(len(df_with_id)))
                df_with_id.to_sql(table_name, self.sqlite_conn, if_exists='replace', index=False)
            else:
                df.to_sql(table_name, self.sqlite_conn, if_exists='replace', index=False)

        self.db = records.Database(f'sqlite:///{self.db_path}')
        self.records_conn = self.db.get_connection()

    def get_table_name(self, table_id: int) -> str:
        table_name = self.table_map.get(table_id)
        if table_name is None:
            raise ValueError(f"Table with id {table_id} not found.")
        return table_name

    # =========================================================================
    # == MODIFIED: execute_query now includes row_id-aware logic =============
    # =========================================================================
    def execute_query(self, sql_query: str, table_name: str) -> Dict[str, Any]:
        """
        Executes a SQL query, intelligently adding 'row_id' to the SELECT statement
        for alignment purposes, based on the reference logic.
        """
        out = None
        sql_query_lower = sql_query.lower().strip()

        # Case 1: Query is just a single column name
        if len(sql_query.split(' ')) == 1 or (sql_query.startswith('`') and sql_query.endswith('`')):
            new_sql_query = f"SELECT row_id, {sql_query} FROM {table_name}"
            out = self.records_conn.query(new_sql_query)
        
        # Case 2: Query already selects all columns or explicitly includes row_id
        elif sql_query_lower.startswith("select *") or "row_id" in sql_query_lower:
            # Replace generic 'w' with the actual table name for execution
            final_sql = sql_query.replace(' w', f' {table_name}')
            out = self.records_conn.query(final_sql)
            
        # Case 3: General SELECT statement, try to add row_id
        else:
            try:
                # Re-write the query to include row_id. Assumes 'SELECT ... FROM w' format.
                select_clause, from_clause = sql_query.split('FROM', 1)
                select_clause = select_clause.replace("SELECT", "", 1).strip()
                new_sql_query = f"SELECT row_id, {select_clause} FROM {table_name}"
                out = self.records_conn.query(new_sql_query)
            except (sqlalchemy.exc.OperationalError, ValueError) as e:
                # Fallback: if rewriting fails, execute the original query
                final_sql = sql_query.replace(' w', f' {table_name}')
                out = self.records_conn.query(final_sql)
        
        if out is None:
            raise RuntimeError("Query execution failed to produce a result.")

        results = out.all()
        headers = out.dataset.headers if out.dataset else []
        rows = [list(r.values()) for r in results]
        return {"header": headers, "rows": rows}
            
    def __del__(self):
        if hasattr(self, 'sqlite_conn'): self.sqlite_conn.close()
        if hasattr(self, 'db_path') and os.path.exists(self.db_path): os.remove(self.db_path)


class Executor(object):
    """Executes SQL queries on a NeuralDB instance."""
    def sql_exec(self, sql: str, db: NeuralDB, table_id: int, verbose=True) -> Dict[str, Any]:
        """
        Passes the user's SQL and target table_id to the enhanced execute_query method.
        """
        table_name = db.get_table_name(table_id)
        if verbose:
            print(f"Executing on table_id {table_id} ('{table_name}'): {sql}")
        
        # Pass the original SQL and the specific table_name to the new execute_query
        # which will now handle all the complex rewriting logic internally.
        result = db.execute_query(sql, table_name)
        return result