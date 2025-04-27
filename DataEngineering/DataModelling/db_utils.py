# db_utils.py - consolidated and cleaned for OLTP & OLAP workflows

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union 
from collections import defaultdict, deque
import re
import os, shutil
import builtins
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import holidays
import warnings
from datetime import datetime, timedelta
from sourceconfig import ETL_CONFIG
from graphviz import Digraph
from IPython.display import display, HTML
import ipywidgets as widgets
import io
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import kaggle
from io import BytesIO
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import sqlite3
from sqlalchemy import create_engine
import pymysql

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


load_dotenv() # Load environment variables from .env file


# ==================================== DB CONNECTION =========================================

# Establish a PostgreSQL connection using environment variables
def get_db_connection(env_prefix: str = "DB_") -> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """
    Establish a PostgreSQL connection and cursor using environment variables with a specified prefix.

    Parameters:
    - env_prefix (str): Prefix for environment variable keys (e.g., 'DB_', 'STAGE_DB_', etc.)

    Required environment variables (with prefix):
    - {PREFIX}USER
    - {PREFIX}PASSWORD
    - {PREFIX}HOST
    - {PREFIX}PORT
    - {PREFIX}NAME

    Returns:
    - tuple containing:
        - psycopg2 connection object
        - psycopg2 cursor object
    - or (None, None) if connection fails
    """
    load_dotenv()

    user = os.getenv(f"{env_prefix}USER")
    password = os.getenv(f"{env_prefix}PASSWORD")
    host = os.getenv(f"{env_prefix}HOST")
    port = os.getenv(f"{env_prefix}PORT")
    database = os.getenv(f"{env_prefix}NAME")

    try:
        conn = psycopg2.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database
        )
        cur = conn.cursor()
        print(f"[get_db_connection] ✅ Connected to '{conn.dsn}' using prefix '{env_prefix}'")
        
        return conn, cur
    except Exception as e:
        print(f"[get_db_connection] ❌ Failed to connect using prefix '{env_prefix}': {e}")
        return None, None

def check_and_create_db(db_params):
    """
    Connect to the default 'postgres' database,
    check if the target database exists,
    and create it if it does not exist.
    Then close the connection.
    """
    try:
        # Connect to default 'postgres' database
        default_db_url = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/postgres"
        conn = psycopg2.connect(default_db_url)
        conn.autocommit = True
        cur = conn.cursor()

        # Check if target database exists
        cur.execute(
            sql.SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s"),
            [db_params['dbname']]
        )
        exists = cur.fetchone()

        if not exists:
            # Create the database
            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(db_params['dbname'])
                )
            )
            print(f"Database '{db_params['dbname']}' created successfully.")
        else:
            print(f"Database '{db_params['dbname']}' already exists.")

        # Close connection to default database
        cur.close()
        conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")

def close_connection(conn, cur):
    """
    Safely close the cursor and connection.
    """
    try:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print("Connection closed successfully.")
    except Exception as e:
        print(f"Error while closing connection: {e}")

def run_sql_script(conn, sql_file_path):
    try:
        with conn.cursor() as cur:
            with open(sql_file_path, 'r') as file:
                sql_script = file.read()
            cur.execute(sql_script)
            conn.commit()
        print(f"Executed SQL script from {sql_file_path} successfully.")
    except Exception as e:
        print(f"Error running SQL script: {e}")

def create_dwh_schema(conn, sql_file_path: str, schemas: dict) -> None:
    """
    Create tables in existing schemas using SQL file.
    
    Args:
        conn: PostgreSQL connection
        sql_file_path: Path to the SQL file containing table definitions
        schemas: Dictionary of schema names to use, e.g. {'oltp_schema': 'public', 'olap_schema': 'zulo_olap'}
    """
    try:
        with conn.cursor() as cur:
            print("\n=== Creating Tables ===")
            with open(sql_file_path, 'r') as f:
                sql_content = f.read()
                
                # Replace schema placeholders with actual schema names
                sql_content = sql_content.format(**schemas)
                
                # Execute each statement
                for statement in sql_content.split(';'):
                    if statement.strip():
                        if 'CREATE TABLE' in statement:
                            # Extract table name for logging
                            parts = statement.split('"')
                            if len(parts) >= 4:
                                schema_name = parts[1]
                                table_name = parts[3]
                                print(f"✅ Creating table: {schema_name}.{table_name}")
                        
                        cur.execute(statement)
            
            conn.commit()
            print("\n=== Table Creation Complete ===")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
        raise

def upsert_from_df(conn, df, table_name: str, schema: str = "zulo_oltp") -> None:
    """
    Simple upsert function for existing tables.
    Uses existing primary key constraints for upsert operations.
    Handles NumPy types, composite keys, and string length validation.
    
    Args:
        conn: PostgreSQL connection
        df: pandas DataFrame containing the data
        table_name: Name of the table to upsert into
        schema: Schema name
    """
    if df.empty:
        print(f"⚠️ DataFrame is empty for table {table_name}")
        return
        
    try:
        with conn.cursor() as cur:
            # Get column information including data types and lengths
            cur.execute(f'''
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = '{schema}'
                AND table_name = '{table_name}';
            ''')
            column_info = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
            
            # Get primary key columns from the table
            cur.execute(f'''
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = '{schema}.{table_name}'::regclass
                AND i.indisprimary;
            ''')
            pk_columns = [row[0] for row in cur.fetchall()]
            
            if not pk_columns:
                raise ValueError(f"No primary key found for table {schema}.{table_name}")
            
            # Get column names from DataFrame
            columns = df.columns.tolist()
            
            # Create placeholders and column names strings
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(f'"{col}"' for col in columns)
            
            # Create the INSERT ... ON CONFLICT query
            pk_str = ', '.join(f'"{col}"' for col in pk_columns)
            update_columns = [col for col in columns if col not in pk_columns]
            
            # For composite keys, we need to handle the ON CONFLICT differently
            if len(pk_columns) > 1:
                # For composite keys, we don't update anything on conflict
                query = f'''
                    INSERT INTO "{schema}"."{table_name}" ({column_names})
                    VALUES ({placeholders})
                    ON CONFLICT ({pk_str}) DO NOTHING
                '''
            else:
                # For single primary key, we can update non-PK columns
                update_str = ', '.join(f'"{col}" = EXCLUDED."{col}"' for col in update_columns)
                query = f'''
                    INSERT INTO "{schema}"."{table_name}" ({column_names})
                    VALUES ({placeholders})
                    ON CONFLICT ({pk_str}) DO UPDATE SET {update_str}
                '''
            
            # Convert DataFrame to list of tuples, handling NumPy types and string lengths
            values = []
            for _, row in df.iterrows():
                row_values = []
                for col, value in row.items():
                    if pd.isna(value):
                        row_values.append(None)
                    elif hasattr(value, 'item'):  # For NumPy types
                        row_values.append(value.item())
                    elif isinstance(value, str) and col in column_info:
                        data_type, max_length = column_info[col]
                        if data_type == 'character varying' and max_length is not None:
                            # Truncate string if it exceeds max length
                            row_values.append(value[:max_length])
                        else:
                            row_values.append(value)
                    else:
                        row_values.append(value)
                values.append(tuple(row_values))
            
            # Execute the query
            cur.executemany(query, values)
            conn.commit()
            
            print(f"✅ Successfully upserted {len(df)} records into {schema}.{table_name}")
            
    except Exception as e:
        print(f"❌ Error upserting records: {e}")
        conn.rollback()
        raise

def upsert_from_file(conn,file_path: str, table_name: str, key_column: str) -> None:
    """
    Update only changed records and insert new ones from CSV.
    """
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        print(f"📖 Read {len(df)} rows from {file_path}")
        
        # Create temp table and load new data
        temp_table = f"temp_{table_name}"
        df.to_sql(temp_table, conn, if_exists='replace', index=False)
        
        # Get column names except key column
        columns = [col for col in df.columns if col != key_column]
        
        # Update only rows that changed
        update_conditions = [
            f"{table_name}.{col} IS NOT {temp_table}.{col}" 
            for col in columns
        ]
        
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE {table_name} 
            SET {', '.join(f'{col} = {temp_table}.{col}' for col in columns)}
            FROM {temp_table}
            WHERE {table_name}.{key_column} = {temp_table}.{key_column}
            AND ({' OR '.join(update_conditions)})
        """)
        updates = cursor.rowcount
        
        # Insert new records
        cursor.execute(f"""
            INSERT INTO {table_name}
            SELECT * FROM {temp_table}
            WHERE {key_column} NOT IN (SELECT {key_column} FROM {table_name})
        """)
        inserts = cursor.rowcount
        
        # Cleanup
        cursor.execute(f"DROP TABLE {temp_table}")
        conn.commit()
        
        print(f"✅ Updated {updates} changed records")
        print(f"✅ Inserted {inserts} new records")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# ==================== 1. Load raw → normalized OLTP tables =========== #

# === DATA READ & EXTRACTION ===

def read_data(source: str, source_type: str = 'auto') -> pd.DataFrame:
    if source_type == 'auto':
        if source.startswith(('http://', 'https://')):
            source_type = 'url'
        elif os.path.isfile(source):
            source_type = 'file'
        elif "/" in source:
            source_type = 'kaggle'
        else:
            source_type = 'gdrive'

    try:
        if source_type == 'file':
            print(f"📂 Loading local file: {source}")
            return pd.read_csv(source)

        elif source_type == 'url':
            print(f"🌐 Loading from URL: {source}")
            return pd.read_csv(source)

        elif source_type == 'gdrive':
            gdrive_url = f"https://drive.google.com/uc?id={source}"
            print(f"🔗 Loading from Google Drive ID: {source}")
            return pd.read_csv(gdrive_url)

        #elif source_type == 'kaggle':
           # return fetch_kaggle_dataset_by_path(source)

    except Exception as e:
        raise ValueError(f"❌ Failed to load data: {e}")
    
def fetch_kaggle_dataset_by_path(path: str, temp_dir=".temp_kaggle") -> dict:
    """
    Downloads a Kaggle dataset using a known path and lets the user pick the file to load.
    Sets `raw_df` and `dataset_key` globally once loaded.
    """
  
    result = {"raw_df": None, "selected_source": None}

    print(f"📦 Loading Kaggle dataset using search: {path}")
    
    os.makedirs(temp_dir, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(path, path=temp_dir, unzip=True)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found.")
        return result

    dropdown = widgets.Dropdown(
        options=csv_files,
        description="Select File:"
    )
    load_button = widgets.Button(description="📥 Load Selected File")
    output = widgets.Output()

    def load_file_callback(_):
        with output:
            selected_file = dropdown.value
            file_path = os.path.join(temp_dir, selected_file)
            try:
                df = pd.read_csv(file_path)
                result["raw_df"] = df
                result["selected_source"] = path

                # Set global vars
                builtins.raw_df = df
                builtins.dataset_key = [k for k, v in ETL_CONFIG["data_sources"].items() if v.get("path") == path][0]
                builtins.cfg = ETL_CONFIG["data_sources"][builtins.dataset_key]

                print(f"✅ Loaded: {selected_file}")
                print("📊 Shape:", df.shape)
                display(df.head(3))
                display(df.info())
                print("\n✅ You can now access raw_df, dataset_key, and cfg globally.")

                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"❌ Failed to load data: {e}")

    load_button.on_click(load_file_callback)
    display(dropdown, load_button, output)
    print("🔄 Please select a file to load.")
   

# ============================= DATA TRANSFORMATION (CLEANING)  =============================================

# 1. Handle Missing Values in Critical Columns
def handle_missing_critical(df: pd.DataFrame, critical_cols: List[str]) -> pd.DataFrame:
    initial_len = len(df)
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    print(f"[handle_missing_critical] Dropped {initial_len - len(df)} rows with nulls in {critical_cols}")
    return df

# 2. Ensure Correct Data Types (including dates)
def ensure_correct_dtypes(df: pd.DataFrame, datetime_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert specified columns to datetime using flexible parsing (mixed formats).
    Safely handles timestamps, day-first and year-first formats, and avoids SettingWithCopyWarning.
    """
    if datetime_cols:
        df = df.copy()  # Ensure no slice issues
        for col in datetime_cols:
            if col in df.columns:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        
                        # Use mixed format inference and normalize
                        parsed = pd.to_datetime(df.loc[:, col], errors="coerce", format="mixed")

                    # Optional: Truncate time to 00:00:00 (pure date)
                    df[col] = parsed.dt.normalize()

                    print(f"[ensure_correct_dtypes] ✅ Converted '{col}' to datetime (mixed format, normalized).")
                except Exception as e:
                    print(f"❌ Failed to convert '{col}' to datetime: {e}")
    return df

# 3. Remove Duplicates Based on Supposed PKs /all  or all columns
def remove_duplicates(df: pd.DataFrame, key_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Removes duplicates based on either specified key columns or all columns.
    """
    before = len(df)
    if key_cols:
        df = df.drop_duplicates(subset=[col for col in key_cols if col in df.columns])
        reason = f"{key_cols}"
    else:
        df = df.drop_duplicates()
        reason = "all columns"
    print(f"[remove_duplicates] Removed {before - len(df)} duplicate row(s) based on {reason}.")
    return df

# 4. Standardize Categorical Fields
def standardize_categoricals(df: pd.DataFrame, categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    print(f"[standardize_categoricals] Standardized fields: {categorical_cols}")
    return df

# 5. Verify Key Relationships (basic null check on FK columns)
def verify_key_relationships(df: pd.DataFrame, fk_cols: List[str]) -> pd.DataFrame:
    for col in fk_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            print(f"[verify_key_relationships] Column '{col}' has {null_count} missing foreign key values.")
    return df

# 6. Split Compound Columns (e.g., Customer_Name → First, Last)
def split_compound_column(df: pd.DataFrame, col: str, new_cols: List[str]) -> pd.DataFrame:
    if col not in df.columns:
        print(f"[split_compound_column] ⚠️ Column '{col}' not found in DataFrame.")
        return df

    max_parts = len(new_cols)
    tokens = df[col].astype(str).str.strip().str.split()

    # Initialize empty target columns
    for new_col in new_cols:
        df[new_col] = ""

    for i, row in df.iterrows():
        parts = tokens[i] if isinstance(tokens[i], list) else []

        if not parts:
            continue

        if len(parts) <= max_parts:
            # Simple left-to-right assignment (e.g., name: First Middle Last)
            for j in range(len(parts)):
                df.at[i, new_cols[j]] = parts[j]
        else:
            # For long strings (e.g., address): assign rightmost parts last → zip/state/city
            for j in range(1, max_parts):
                df.at[i, new_cols[-j]] = parts[-j]
            df.at[i, new_cols[0]] = " ".join(parts[:-max_parts + 1])

    print(f"[split_compound_column] ✅ Split '{col}' into {new_cols}")
    return df

# 7. Validate Contact Info (basic cleaning)
def validate_contact_info(df: pd.DataFrame, email_col: Optional[str] = None, phone_col: Optional[str] = None) -> pd.DataFrame:
    if email_col and email_col in df.columns:
        df[email_col] = df[email_col].astype(str).str.strip().str.lower()
        print(f"[validate_contact_info] Cleaned email in column '{email_col}'")
    if phone_col and phone_col in df.columns:
        df[phone_col] = df[phone_col].astype(str).str.replace(r'[^\d+]', '', regex=True)
        print(f"[validate_contact_info] Cleaned phone in column '{phone_col}'")
    return df

# 8. Normalize Derived Fields (e.g., recompute total = price * quantity)
def normalize_derived_fields(df: pd.DataFrame, price_col: str, qty_col: str, new_col: str = 'computed_total') -> pd.DataFrame:
    """
    Create a derived column (e.g., total = price * quantity) and round to 2 decimal places.

    Parameters:
    - df: input DataFrame
    - price_col: name of the price column
    - qty_col: name of the quantity column
    - new_col: name of the new derived column to create (default: 'computed_total')

    Returns:
    - Updated DataFrame with the derived column
    """
    if price_col in df.columns and qty_col in df.columns:
        df[new_col] = (df[price_col] * df[qty_col]).round(2)
        print(f"[normalize_derived_fields] Created '{new_col}' from {price_col} * {qty_col}, rounded to 2dp.")
    return df

# 9. Standardize Column Names to snake_case
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names:
    - If the column has underscores, assume it's already snake_case and just lowercase it.
    - Otherwise, convert camelCase, PascalCase, or space-separated to snake_case.
    """
    def clean_name(name: str) -> str:
        if '_' in name:
            return name.lower()
        # Convert to snake_case
        name = re.sub(r"[\s\-]+", "_", name)  # Replace spaces and hyphens with underscore
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)  # camelCase to snake
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)  # PascalCase to snake
        name = re.sub(r"__+", "_", name)  # Collapse multiple underscores
        name = re.sub(r"[^\w_]", "", name)  # Remove non-alphanumeric

        return name.lower()

    df.columns = [clean_name(col) for col in df.columns]
    return df

# 10. Compute Interest based on principal and rate columns
def compute_interest(df: pd.DataFrame, principal_col: str, rate_col: str, new_col: str = 'computed_interest') -> pd.DataFrame:
    """
    Compute simple interest and add as a new column.
    Assumes simple interest: Interest = Principal × Rate
    """
    if principal_col in df.columns and rate_col in df.columns:
        df[new_col] = (df[principal_col] * df[rate_col] / 100).round(2)
        print(f"[compute_interest] ✅ Computed '{new_col}' from {principal_col} * {rate_col} / 100")
    return df

# helper ETL function for processing a single data source
def process_data_source(df: pd.DataFrame, source_name: str, source_config: dict) -> Tuple[str, pd.DataFrame]:
    """
    Apply standard preprocessing pipeline based on source config.
    Returns cleaned dataframe.
    """
    print(f"\n🧪 Processing source: {source_name}")
    
    df = ensure_correct_dtypes(df, source_config.get("date_columns"))

     # 3. Core transformations — use optional fallbacks
    if "missingvalue_columns" in source_config:
        df = handle_missing_critical(df, source_config["missingvalue_columns"])
        
     # ✅ Use the flexible duplicate remover — all columns by default
    #df = remove_duplicates(df)
  
    # 2. Optional: Split compound name fields
    for col, new_cols in source_config.get("split_columns", {}).items():
        if col in df.columns:
            df = split_compound_column(df, col, new_cols)

    if "contact_columns" in source_config:
        df = validate_contact_info(df, **source_config["contact_columns"])

    if "derived_fields" in source_config:
        df = normalize_derived_fields(df, **source_config["derived_fields"])
    
    df = standardize_categoricals(df)

    # 4. Final standardization
    df = standardize_column_names(df)

    print(f"✅ Finished processing: {source_name} — {df.shape[0]} rows, {df.shape[1]} columns")
    return source_name, df


# =========================  GET NORMALIZED OLTP TABLES post sort =======================
def topological_sort_tables(
    tables: Dict[str, pd.DataFrame],
    foreign_keys: Dict[str, List[Tuple[str, str, str]]]
) -> List[str]:
    """
    Performs topological sorting of tables based on FK dependencies.

    Correctly builds the graph such that a table appears only after the tables
    it depends on (i.e., those it references via foreign keys).

    Parameters:
    - tables: dict of table_name -> DataFrame
    - foreign_keys: dict of table_name -> list of (fk_col, ref_table, ref_col)

    Returns:
    - sorted list of table names respecting FK dependency order
    """
    graph = defaultdict(set)       # key: table, value: set of tables it points to (depends on)
    in_degree = defaultdict(int)   # key: table, value: number of incoming dependencies

    # Initialize graph
    for table in tables:
        graph[table] = set()
        in_degree[table] = 0

    # Build edges: from_table depends on to_table
    for from_table, fks in foreign_keys.items():
        for _, to_table, _ in fks:
            if to_table not in graph[from_table]:
                graph[from_table].add(to_table)
                in_degree[to_table] += 1

    # Topological sort (Kahn's algorithm)
    queue = deque([t for t in tables if in_degree[t] == 0])
    sorted_tables = []

    while queue:
        node = queue.popleft()
        sorted_tables.append(node)
        for dependent in graph[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_tables) != len(tables):
        raise ValueError("Cycle detected or missing dependencies in FK graph.")

    return sorted_tables

def split_normalized_tables(
    df: pd.DataFrame,
    table_specs: Dict[str, List[str]],
    critical_columns: Dict[str, List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Splits raw DataFrame into normalized tables based on table specs.
    Only drops rows with nulls in critical columns, and deduplicates full records only.
    """
    critical_columns = critical_columns or {}
    tables = {}

    print("\n🧩 Splitting and normalizing tables...")
    for table, cols in table_specs.items():
        selected = [c for c in cols if c in df.columns]
        if not selected:
            print(f"⚠️ No matching columns for '{table}' — skipping.")
            continue

        table_df = df[selected].copy()
        original_count = len(table_df)

        # Drop rows with nulls in any critical columns
        crits = [c for c in critical_columns.get(table, []) if c in table_df.columns]
        if crits:
            before_null = len(table_df)
            table_df = table_df.dropna(subset=crits)
            after_null = len(table_df)
            print(f"🔍 '{table}': Dropped {before_null - after_null} rows with nulls in {crits}")

        # Drop FULL duplicate rows only (no subset deduping)
        before_dedup = len(table_df)
        table_df = table_df.drop_duplicates().reset_index(drop=True)
        after_dedup = len(table_df)
        print(f"🧹 '{table}': Removed {before_dedup - after_dedup} full duplicate rows")

        print(f"✅ Created '{table}': {after_dedup} rows (from {original_count})")
        tables[table] = table_df

    return tables

# =========================== checks if pk exists in splitted tables & creates if it doesnt === #
def generate_index_pks(
    df: pd.DataFrame,
    table_mapping: Dict[str, List[str]],
    pk: dict = None,
    critical_columns: dict = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Enforces exactly one primary key per table in the format <table>_id.
    Treats all other *_id fields as foreign keys.

    - Uses critical column if unique
    - Falls back to existing <table>_id if unique
    - Otherwise generates <table>_id as synthetic key

    Returns:
    - updated_df: modified DataFrame with guaranteed PK column
    - pk_dict: {table_name: [<table>_id]}
    """
    pk = pk or {}
    critical_columns = critical_columns or {}
    updated_df = df.copy()

    for table_name, columns in table_mapping.items():
        if table_name in pk and pk[table_name]:
            continue

        singular = table_name[:-1] if table_name.endswith("s") else table_name
        enforced_pk_col = f"{singular}_id"

        # Special case: date_dim → date_id
        if table_name == "date_dim" and "date_id" in updated_df.columns and updated_df["date_id"].is_unique:
            pk[table_name] = ["date_id"]
            print(f"✅ Inferred PK for 'date_dim': date_id")
            continue

        # Priority 1: critical columns
        criticals = critical_columns.get(table_name, [])
        for col in criticals:
            if col in updated_df.columns and updated_df[col].is_unique:
                if col != enforced_pk_col:
                    updated_df[enforced_pk_col] = updated_df[col]
                    print(f"🧭 Copied PK from {col} → {enforced_pk_col}")
                pk[table_name] = [enforced_pk_col]
                break

        # Priority 2: existing enforced_pk_col if already unique
        if table_name not in pk:
            if enforced_pk_col in updated_df.columns and updated_df[enforced_pk_col].is_unique:
                pk[table_name] = [enforced_pk_col]
                print(f"✅ Using existing {enforced_pk_col} as PK for {table_name}")
            else:
                # Priority 3: generate synthetic key
                print(f"⚠️ Generating synthetic PK '{enforced_pk_col}' for '{table_name}'")
                updated_df = updated_df.reset_index(drop=True)
                updated_df[enforced_pk_col] = range(1, len(updated_df) + 1)
                updated_df[enforced_pk_col] = updated_df[enforced_pk_col].astype("Int64")
                pk[table_name] = [enforced_pk_col]

    return updated_df, pk

def enforce_foreign_keys(
    temp_tables: Dict[str, pd.DataFrame],
    pk_dict: Dict[str, List[str]],
    foreign_keys: Union[List[Dict[str, str]], Dict[str, List[Tuple[str, str, str]]]],
    return_fk_dict: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[Tuple[str, str, str]]]]:
    """
    Enforces FK rules using config or tuple-based FK dictionary.
    Adds FK placeholder columns to source tables and builds a normalized fk_dict.
    
    Supports:
    - foreign_keys as a list of dicts (from config)
    - or as a dict of (fk_col, ref_table, ref_col) tuples

    Returns:
    - updated temp_tables
    - fk_dict: {from_table: [(fk_col, to_table, to_col)]}
    """
    fk_dict = {}

    if isinstance(foreign_keys, dict):
        # Already normalized tuple-based format
        for from_tbl, fk_list in foreign_keys.items():
            for fk_col, to_tbl, to_col in fk_list:
                if from_tbl == to_tbl and fk_col in pk_dict.get(from_tbl, []):
                    print(f"⏩ Skipping self-referencing PK FK: {from_tbl}.{fk_col}")
                    continue

                if fk_col not in temp_tables.get(from_tbl, {}).columns:
                    temp_tables[from_tbl][fk_col] = pd.Series(dtype="Int64")
                    print(f"➕ FK column '{fk_col}' added to '{from_tbl}'")

                fk_dict.setdefault(from_tbl, []).append((fk_col, to_tbl, to_col))
    else:
        # List of config-style FK dicts
        for fk in foreign_keys:
            from_tbl = fk.get("from")
            to_tbl = fk.get("to")
            from_col = fk.get("source_column")
            to_col = fk.get("target_column")

            if not all([from_tbl, to_tbl, from_col, to_col]):
                print(f"⚠️ FK skipped — missing fields: {fk}")
                continue

            if from_tbl == to_tbl and from_col in pk_dict.get(from_tbl, []):
                print(f"⏩ Skipping self-FK {from_tbl}.{from_col} — it's the table's PK.")
                continue

            if from_tbl not in temp_tables or to_tbl not in temp_tables:
                print(f"⚠️ FK skipped — table missing: {from_tbl} or {to_tbl}")
                continue

            if from_col not in temp_tables[from_tbl].columns:
                temp_tables[from_tbl][from_col] = pd.Series(dtype="Int64")
                print(f"➕ FK column '{from_col}' added to '{from_tbl}'")

            fk_dict.setdefault(from_tbl, []).append((from_col, to_tbl, to_col))

    return (temp_tables, fk_dict) if return_fk_dict else temp_tables

# Convert PK columns to Int64 if numeric but not integers
def convert_pk_column_to_int(tables: Dict[str, pd.DataFrame], pk_dict: Dict[str, List[str]]):
    """
    Converts PK columns to Int64 if they're numeric but not integers.
    """
    for table, pk_cols in pk_dict.items():
        for pk in pk_cols:
            if pk in tables[table].columns:
                col_dtype = tables[table][pk].dtype
                if pd.api.types.is_float_dtype(col_dtype) or pd.api.types.is_numeric_dtype(col_dtype):
                    tables[table][pk] = pd.to_numeric(tables[table][pk], errors='coerce').astype("Int64")
                    print(f"🔢 Converted '{table}.{pk}' to Int64")

# ============================================= 2. Build and populate dim_date (DW:IF OLAP CONSIDERED)================================ # 
def extract_date_columns_from_temp(
    temp_tables: Dict[str, pd.DataFrame],
    date_columns: List[str]
) -> pd.DataFrame:
    """
    Extracts available date columns from already standardized temp_tables (OLTP tables).
    Only uses columns present in the tables (no fuzzy matching).
    Returns a combined DataFrame for use in date_dim generation.
    """
    if not date_columns:
        print("[extract_date_columns_from_oltp] ⚠️ No date_columns provided in config.")
        return pd.DataFrame()

    candidate_frames = []
    found_cols = set()

    for table_name, df in temp_tables.items():
        matching_cols = [col for col in date_columns if col in df.columns]
        if matching_cols:
            found_cols.update(matching_cols)
            candidate_frames.append(df[matching_cols])

    missing_cols = set(date_columns) - found_cols
    if missing_cols:
        print(f"[extract_date_columns_from_oltp] ⚠️ The following date columns were not found in any table: {sorted(missing_cols)}")

    if candidate_frames:
        return pd.concat(candidate_frames, axis=0, ignore_index=True)

    print("[extract_date_columns_from_oltp] ⚠️ No matching date columns found across temp_tables.")
    return pd.DataFrame()

def generate_date_dim(
    df: pd.DataFrame,
    date_columns: List[str],
    id_col: str = 'date_id',
    country: str = 'US'
) -> pd.DataFrame:
    """
    Generate a date dimension DataFrame from one or more date columns, including holiday flag.
    Supports mixed formats (e.g., with time components).
    """
    all_dates = pd.Series(dtype='datetime64[ns]')
    for col in date_columns:
        if col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                all_dates = pd.concat([all_dates, parsed.dropna()])
            except Exception as e:
                print(f"[generate_date_dim] ⚠️ Failed to parse column '{col}': {e}")

    # Drop duplicates and normalize to remove time
    unique_dates = all_dates.dropna().drop_duplicates().sort_values().dt.normalize()

    if unique_dates.empty:
        print("[generate_date_dim] ⚠️ No valid dates found after parsing.")
        return pd.DataFrame()

    date_dim = pd.DataFrame({'full_date': unique_dates})
    date_dim[id_col] = date_dim['full_date'].dt.strftime('%Y%m%d').astype(int)
    date_dim['day'] = date_dim['full_date'].dt.day
    date_dim['month'] = date_dim['full_date'].dt.month
    date_dim['year'] = date_dim['full_date'].dt.year
    date_dim['quarter'] = date_dim['full_date'].dt.quarter
    date_dim['week'] = date_dim['full_date'].dt.isocalendar().week
    date_dim['day_name'] = date_dim['full_date'].dt.day_name()
    date_dim['month_name'] = date_dim['full_date'].dt.month_name()
    date_dim['is_weekend'] = date_dim['full_date'].dt.weekday >= 5

    try:
        import holidays
        country_holidays = holidays.country_holidays(country)
        date_dim['is_holiday'] = date_dim['full_date'].isin(country_holidays)
    except Exception as e:
        print(f"[generate_date_dim] ⚠️ Could not determine holidays for '{country}': {e}")
        date_dim['is_holiday'] = False

    print(f"[generate_date_dim] ✅ Created date dimension with {len(date_dim)} unique dates.")
    return date_dim

def apply_configured_date_mapping(
    tables: Dict[str, pd.DataFrame],
    date_dim: pd.DataFrame,
    date_mapping: Dict[str, str],
    date_key: str = "date_id"
) -> Dict[str, pd.DataFrame]:
    """
    Maps *_date columns in tables to date_dim[date_id] using the provided mapping config.
    Returns updated tables with *_date_id columns added.
    """
    date_dim_lookup = date_dim[["full_date", date_key]].drop_duplicates()

    for raw_col, mapped_col in date_mapping.items():
        raw_col_lower = raw_col.lower()

        for tbl_name, df in tables.items():
            if raw_col_lower in df.columns:
                print(f"🔗 Mapping {raw_col_lower} → {mapped_col} in table '{tbl_name}'")

                # ⛳ FIX: Convert the column to datetime to match date_dim
                df[raw_col_lower] = pd.to_datetime(df[raw_col_lower], errors="coerce").dt.normalize()

                # Perform the join
                df = df.merge(date_dim_lookup, how="left", left_on=raw_col_lower, right_on="full_date")
                df = df.drop(columns=["full_date"]).rename(columns={date_key: mapped_col})
                df[mapped_col] = df[mapped_col].astype("Int64")  # nullable FK

                tables[tbl_name] = df

    return tables

def copy_date_dim_to_olap(conn, oltp_tables: Dict[str, pd.DataFrame], olap_schema: str):
    """
    Copies 'date_dim' from OLTP to OLAP schema using existing schema from OLTP.
    Skips creation if already exists and safely inserts rows.
    """
    table_name = "date_dim"
    df = oltp_tables.get(table_name)

    if df is None or df.empty:
        print(f"⚠️ Skipping copy — '{table_name}' not found or empty in OLTP.")
        return

    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            );
        """, (olap_schema, table_name))
        exists = cur.fetchone()[0]

        if not exists:
            cur.execute(f'CREATE TABLE "{olap_schema}"."{table_name}" (LIKE zulo_oltp."{table_name}" INCLUDING ALL);')
            print(f"🧱 Created '{olap_schema}.{table_name}' using OLTP structure.")

        conn.commit()

    # Upsert into OLAP schema using PK from OLTP
    upsert_dataframe_to_table(
        conn,
        df,
        table_name=table_name,
        schema=olap_schema,
        pk_cols=["date_id"],
        skip_on_fk_violation=True
    )

# ================================================= CREATE SCHEMA & TABLES (OLTP)==== #

def generate_sql_type(dtype: str) -> str:
    dtype = dtype.lower()
    if "int" in dtype:
        return "INT"
    elif "float" in dtype or "double" in dtype:
        return "DOUBLE PRECISION"
    elif "bool" in dtype:
        return "BOOLEAN"
    elif "datetime" in dtype or "timestamp" in dtype:
        return "TIMESTAMP"
    elif "date" in dtype:
        return "DATE"
    elif "object" in dtype or "string" in dtype or "category" in dtype:
        return "TEXT"
    return "TEXT"

def create_tables_without_foreign_keys(
    conn,
    schema: str,
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, List[str]]
) -> None:
    cur = conn.cursor()

    # Ensure schema exists
    cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s", (schema,))
    if not cur.fetchone():
        cur.execute(f'CREATE SCHEMA "{schema}"')
        print(f"📁 Created schema: {schema}")

    for table_name, df in tables.items():
        # Skip if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            )
        """, (schema, table_name))
        if cur.fetchone()[0]:
            print(f"⏩ {schema}.{table_name} exists — skipping")
            continue

        print(f"🧱 Creating {schema}.{table_name}...")

        cols = []
        for col in df.columns:
            sql_type = generate_sql_type(str(df[col].dtype))
            cols.append(f'"{col}" {sql_type}')

        pk = primary_keys.get(table_name, [])
        if pk:
            quoted_pk = ', '.join([f'"{c}"' for c in pk])
            cols.append(f'PRIMARY KEY ({quoted_pk})')

        create_sql = f'CREATE TABLE "{schema}"."{table_name}" (\n  ' + ',\n  '.join(cols) + '\n)'
        cur.execute(create_sql)
        print(f"✅ Created table: {table_name}")

    conn.commit()
    cur.close()

def apply_foreign_keys(
    conn,
    schema: str,
    foreign_keys: Dict[str, List[Tuple[str, str, str]]]
) -> None:
    """
    Adds FK constraints using ALTER TABLE — only if referenced column is UNIQUE or PK.
    """
    cur = conn.cursor()

    for from_table, fks in foreign_keys.items():
        for fk_col, to_table, to_col in fks:
            constraint_name = f"fk_{from_table}_{fk_col}_to_{to_table}_{to_col}"

            # Skip if already applied
            cur.execute("""
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_schema = %s AND table_name = %s AND constraint_name = %s
            """, (schema, from_table, constraint_name))
            if cur.fetchone():
                print(f"⏩ FK exists: {constraint_name} — skipping")
                continue

            # Ensure referenced column is unique or PK
            cur.execute("""
                SELECT 1
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                 AND tc.constraint_schema = ccu.constraint_schema
                WHERE tc.table_schema = %s
                  AND tc.table_name = %s
                  AND ccu.column_name = %s
                  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
            """, (schema, to_table, to_col))
            if not cur.fetchone():
                print(f"⚠️ Skipped FK: {from_table}.{fk_col} → {to_table}.{to_col} — target not PK or UNIQUE")
                continue

            try:
                cur.execute(f'''
                    ALTER TABLE "{schema}"."{from_table}"
                    ADD CONSTRAINT "{constraint_name}"
                    FOREIGN KEY ("{fk_col}")
                    REFERENCES "{schema}"."{to_table}" ("{to_col}");
                ''')
                print(f"🔗 Enforced FK: {from_table}.{fk_col} → {to_table}.{to_col}")
            except Exception as e:
                print(f"❌ FK failed: {from_table}.{fk_col} → {to_table}.{to_col} — {e}")

    conn.commit()
    cur.close()

def create_oltp_schema_and_tables(
    conn,
    schema: str,
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, List[str]],
    foreign_keys: Dict[str, List[Tuple[str, str, str]]]
) -> None:
    print(f"📦 Creating OLTP schema: {schema}")
    create_tables_without_foreign_keys(conn, schema, tables, primary_keys)
    apply_foreign_keys(conn, schema, foreign_keys)
    print("✅ OLTP schema and constraints applied.\n")

def display_database_info(oltp_tables, pk_dict, fk_dict, sk_dict):
    # Display key information
    print("\n=== Database Structure ===")
    print(f"Tables: {list(oltp_tables.keys())}")
    print("\n")
    print(f"\nPrimary Keys: {pk_dict}")
    print("\n")
    print(f"Foreign Keys: {fk_dict}")
    print("\n")
    print(f"Surrogate Keys: {sk_dict}")
    
    # Display table details
    print("\n=== Table Details ===")
    for name, df in oltp_tables.items():
        print(f"\n📊 {name.upper()}")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print("\nSample Data:")
        display(df.head(3))
        print("\nColumn Info:")
        display(df.dtypes)
        print("-" * 50)

# ==== Load Data into OLTP Tables (staging) ====#Ensure all FKs (customer_id, account_id, *_date_id) are present and non-null where expected
def extract_fact_columns(field_config: List[Union[str, Dict[str, str]]]) -> List[str]:
    """
    Extracts flat column names from a fact field config.
    """
    cols = []
    for item in field_config:
        if isinstance(item, str):
            cols.append(item)
        elif isinstance(item, dict):
            cols.extend(item.keys())
    return cols

def upsert_dataframe_to_table(
    conn,
    df: pd.DataFrame,
    table_name: str,
    schema: str = "public",
    pk_cols: Optional[List[str]] = None,
    skip_on_fk_violation: bool = False  # 🔄 NEW FLAG
) -> None:
    import io
    from psycopg2.errors import ForeignKeyViolation
    import psycopg2

    if df.empty:
        print(f"[UPSERT] ⚠️ Skipping '{schema}.{table_name}' — empty DataFrame")
        return

    try:
        cur = conn.cursor()

        # Step 1: Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s 
                AND table_name = %s
            );
        """, (schema, table_name))
        exists = cur.fetchone()[0]

        if not exists:
            print(f"[UPSERT] ❌ Table '{schema}.{table_name}' does not exist — cannot insert")
            return

        # Step 2: Fetch DB column info
        cur.execute("""
            SELECT column_name, is_nullable, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """, (schema, table_name))
        column_info = cur.fetchall()
        db_columns = [col[0] for col in column_info]
        column_nullability = {col[0]: col[1] for col in column_info}
        column_types = {col[0]: col[2] for col in column_info}

        # Step 3: Clean DataFrame
        cleaned_df = df.copy()
        for col in db_columns:
            if col not in cleaned_df.columns:
                if column_nullability[col] == 'NO':
                    if 'int' in column_types[col]:
                        cleaned_df[col] = -1
                    elif 'date' in column_types[col]:
                        cleaned_df[col] = pd.NaT
                    elif 'double' in column_types[col] or 'numeric' in column_types[col]:
                        cleaned_df[col] = 0.0
                    else:
                        cleaned_df[col] = ''
                else:
                    cleaned_df[col] = None

        # Fill non-nullable string columns
        for col in db_columns:
            if column_nullability[col] == 'NO' and col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].replace('', 'unknown').fillna('unknown')

        cleaned_df = cleaned_df[db_columns]

        if skip_on_fk_violation:
            # 🔄 Insert row by row with FK violation handling
            inserted, skipped = 0, 0
            for i, row in cleaned_df.iterrows():
                placeholders = ", ".join(["%s"] * len(db_columns))
                insert_cols = ', '.join(f'"{c}"' for c in db_columns)
                conflict_cols = ', '.join(f'"{c}"' for c in pk_cols or [])
                updates = ', '.join(f'"{c}" = EXCLUDED."{c}"' for c in db_columns if not pk_cols or c not in pk_cols)

                insert_sql = f"""
                    INSERT INTO "{schema}"."{table_name}" ({insert_cols})
                    VALUES ({placeholders})
                    ON CONFLICT ({conflict_cols})
                    DO UPDATE SET {updates};
                """ if pk_cols else f"""
                    INSERT INTO "{schema}"."{table_name}" ({insert_cols})
                    VALUES ({placeholders})
                    ON CONFLICT DO NOTHING;
                """

                try:
                    cur.execute(insert_sql, tuple(row))
                    inserted += 1
                except ForeignKeyViolation:
                    conn.rollback()  # must rollback failed tx
                    skipped += 1
                except Exception as e:
                    conn.rollback()
                    print(f"[ROW FAIL] ❌ Row {i}: {e}")
                    skipped += 1

            conn.commit()
            print(f"[UPSERT] ✅ Inserted {inserted} rows into {schema}.{table_name}")
            if skipped:
                print(f"⚠️ Skipped {skipped} rows due to FK violations")
        else:
            # 🚀 Bulk insert to temp table (faster but all-or-nothing)
            buffer = io.StringIO()
            cleaned_df.to_csv(buffer, index=False, header=False, sep="\t", na_rep="\\N")
            buffer.seek(0)

            temp_table = f"temp_{table_name}"
            cur.execute(f'DROP TABLE IF EXISTS "{temp_table}"')
            cur.execute(f'CREATE TEMP TABLE "{temp_table}" (LIKE "{schema}"."{table_name}" INCLUDING ALL);')
            cur.copy_from(buffer, temp_table, sep="\t", null="\\N", columns=db_columns)

            insert_columns = ', '.join(f'"{col}"' for col in db_columns)
            update_assignments = ', '.join(f'"{col}" = EXCLUDED."{col}"' for col in db_columns if pk_cols and col not in pk_cols)

            if pk_cols:
                conflict_cols = ', '.join(f'"{col}"' for col in pk_cols)
                upsert_sql = f'''
                    INSERT INTO "{schema}"."{table_name}" ({insert_columns})
                    SELECT {insert_columns} FROM "{temp_table}"
                    ON CONFLICT ({conflict_cols})
                    DO UPDATE SET {update_assignments};
                '''
            else:
                upsert_sql = f'''
                    INSERT INTO "{schema}"."{table_name}" ({insert_columns})
                    SELECT {insert_columns} FROM "{temp_table}"
                    ON CONFLICT DO NOTHING;
                '''

            cur.execute(upsert_sql)
            conn.commit()
            print(f"[UPSERT] ✅ Successfully upserted {len(cleaned_df)} rows to {schema}.{table_name}")

    except Exception as e:
        conn.rollback()
        print(f"[UPSERT] ❌ Failed to upsert '{schema}.{table_name}': {e}")
        import logging
        logging.error(f"[UPSERT] ❌ Error: {e}")
    finally:
        if 'cur' in locals():
            cur.close()

# ============================== OLAP WORKFLOW  FROM OLTP :  ========================== #
def create_table_if_not_exists(
    conn, 
    df: pd.DataFrame, 
    table_name: str, 
    schema: str = "public",
    pk_cols: List[str] = None,
    fk_constraints: List[Dict] = None,
    is_olap: bool = False
) -> None:
    """
    Create table with proper constraints for both OLTP and OLAP schemas.
    """
    def get_sql_type(col_name: str, dtype, is_olap: bool) -> str:
        if col_name.endswith('_id') and not is_olap:
            return 'SERIAL' if col_name in (pk_cols or []) else 'INTEGER'
        if col_name.endswith('_sk') and is_olap:
            return 'INTEGER'
        if col_name.endswith('date_id') and is_olap:
            return 'INTEGER'
            
        if np.issubdtype(dtype, np.integer):
            return 'INTEGER'
        elif np.issubdtype(dtype, np.floating):
            return 'NUMERIC'
        elif np.issubdtype(dtype, np.datetime64):
            return 'TIMESTAMP'
        elif np.issubdtype(dtype, np.bool_):
            return 'BOOLEAN'
        else:
            if dtype == object:
                max_length = df[col_name].astype(str).str.len().max()
                if max_length <= 255:
                    return f'VARCHAR({max_length})'
            return 'TEXT'

    # Generate column definitions
    col_defs = []
    for col in df.columns:
        sql_type = get_sql_type(col, df[col].dtype, is_olap)
        nullable = 'NOT NULL' if col in (pk_cols or []) else ''
        col_defs.append(f'"{col}" {sql_type} {nullable}')

    # Add constraints
    constraints = []
    
    # Primary Key constraint
    if pk_cols:
        pk_cols_quoted = ', '.join(f'"{c}"' for c in pk_cols)
        pk_constraint = f'CONSTRAINT "{table_name}_pkey" PRIMARY KEY ({pk_cols_quoted})'
        constraints.append(pk_constraint)

    # Foreign Key constraints
    if fk_constraints:
        for fk in fk_constraints:
            constraint_name = f'fk_{table_name}_{fk["column"]}_to_{fk["references"]["table"]}'
            fk_constraint = (
                f'CONSTRAINT "{constraint_name}" '
                f'FOREIGN KEY ("{fk["column"]}") '
                f'REFERENCES "{schema}"."{fk["references"]["table"]}" '
                f'("{fk["references"]["column"]}")'
            )
            constraints.append(fk_constraint)

    # Build the CREATE TABLE statement using string concatenation to avoid backslash issues
    create_stmt = (
        f'CREATE TABLE IF NOT EXISTS "{schema}"."{table_name}" (\n'
        + ',\n    '.join(col_defs)
    )
    
    # Add constraints if they exist
    if constraints:
        create_stmt += ',\n    ' + ',\n    '.join(constraints)
    
    # Close the statement
    create_stmt += '\n)'

    with conn.cursor() as cur:
        try:
            cur.execute(create_stmt)
            conn.commit()
            print(f"[CREATE] 🧱 Created table '{schema}.{table_name}'")
            if pk_cols:
                print(f"[CREATE] 🔑 Primary Key: {pk_cols}")
            if fk_constraints:
                for fk in fk_constraints:
                    print(f"[CREATE] 🔗 Foreign Key: {fk['column']} → {fk['references']['table']}.{fk['references']['column']}")
        except Exception as e:
            print(f"[CREATE] ❌ Failed to create table: {str(e)}")
            conn.rollback()
            raise

def build_and_load_all_dimensions(
    conn,
    oltp_tables: Dict[str, pd.DataFrame],
    dimension_defs: Dict[str, List[str]],
    schema: str = "public"
) -> Dict[str, pd.DataFrame]:
    """
    Builds and loads all dimension tables using surrogate keys added during OLTP transformation.
    Reuses *_sk columns and ensures they're included as primary keys in dimension tables.
    Returns a lookup dict of business keys + SKs per dimension.
    """
    lookup_tables = {}

    for dim_table, columns in dimension_defs.items():
        source_table = dim_table.replace("dim_", "")
        if source_table not in oltp_tables:
            print(f"[DIM LOAD] ⚠️ Source '{source_table}' not found — skipping '{dim_table}'")
            continue

        df = oltp_tables[source_table].copy()
        sk_col = f"{source_table}_sk"

        if sk_col not in df.columns:
            print(f"[DIM LOAD] ❌ SK column '{sk_col}' missing in '{source_table}' — skipping")
            continue

        if not all(col in df.columns for col in columns):
            print(f"[DIM LOAD] ❌ Missing business keys for '{dim_table}' — skipping")
            continue

        dim_df = df[[sk_col] + columns].drop_duplicates().reset_index(drop=True)

        # Create dimension table with PK if not exists
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                );
            """, (schema, dim_table))
            exists = cur.fetchone()[0]

            if not exists:
                col_defs = [f'"{sk_col}" INT PRIMARY KEY'] + [f'"{col}" TEXT' for col in columns]
                cur.execute(f'CREATE TABLE "{schema}"."{dim_table}" ({", ".join(col_defs)});')
                print(f"[DIM CREATE] 🧱 Created '{schema}.{dim_table}' with PK '{sk_col}'")
            conn.commit()

        upsert_dataframe_to_table(
            conn,
            dim_df,
            table_name=dim_table,
            schema=schema,
            pk_cols=[sk_col]
        )
        print(f"[DIM LOAD] ✅ Loaded {len(dim_df)} rows into '{schema}.{dim_table}'")

        # Prepare lookup for fact joins
        business_keys = [c for c in columns if c.endswith("_id")]
        lookup_tables[dim_table] = dim_df[business_keys + [sk_col]]

    return lookup_tables

def build_and_load_all_facts(
    conn,
    oltp_tables: Dict[str, pd.DataFrame],
    fact_config: Dict[str, Dict],
    dim_lookups: Dict[str, pd.DataFrame],
    schema: str = "public"
) -> Dict[str, pd.DataFrame]:
    fact_lookups = {}
    
    for fact_name, config in fact_config.items():
        print(f"\n📦 Building '{fact_name}'")
        
        # Get base fact table
        source_table = config.get("source_table") or fact_name.replace("fact_", "")
        if source_table not in oltp_tables:
            print(f"[FACT LOAD] ⚠️ Source '{source_table}' not found")
            continue

        # Start with base table
        fact_df = oltp_tables[source_table].copy()

        # Apply joins first
        for join_config in config.get("joins", []):
            try:
                table_name, join_key, columns = join_config
                if table_name not in oltp_tables:
                    print(f"[FACT JOIN] ⚠️ Table '{table_name}' not found")
                    continue

                join_cols = [join_key] + [col for col in columns if col not in fact_df.columns]
                join_df = oltp_tables[table_name][join_cols].drop_duplicates()
                
                fact_df = fact_df.merge(
                    join_df,
                    on=join_key,
                    how="left"
                )
                print(f"[FACT JOIN] Joined '{table_name}' on '{join_key}' → added: {columns}")

            except Exception as e:
                print(f"[FACT JOIN] ❌ Join failed: {e}")
                continue

        # Map surrogate keys from dimensions
        for dim_name, dim_df in dim_lookups.items():
            try:
                base_name = dim_name.replace("dim_", "")
                natural_key = f"{base_name}_id"
                surrogate_key = f"{base_name}_sk"
                
                if natural_key in fact_df.columns:
                    sk_mapping = dim_df[[natural_key, surrogate_key]].drop_duplicates()
                    
                    if surrogate_key in fact_df.columns:
                        fact_df = fact_df.drop(columns=[surrogate_key])
                    
                    fact_df = fact_df.merge(
                        sk_mapping,
                        on=natural_key,
                        how="left",
                        validate="many_to_one"
                    )
                    
                    missing_count = fact_df[surrogate_key].isna().sum()
                    if missing_count > 0:
                        print(f"[FACT JOIN] ⚠️ Warning: {missing_count} rows have missing {surrogate_key}")
                    else:
                        print(f"[FACT JOIN] Mapped {natural_key} → {surrogate_key}")
                        
            except Exception as e:
                print(f"[FACT JOIN] ❌ SK mapping failed for {dim_name}: {e}")
                continue

        # Handle calculated fields
        for field in config.get("fields", []):
            if isinstance(field, dict):
                for new_col, expr in field.items():
                    try:
                        fact_df[new_col] = pd.eval(expr, local_dict=fact_df)
                        fact_df[new_col] = fact_df[new_col].round(2)
                        print(f"[FACT BUILD] 🧮 Calculated '{new_col}' using '{expr}'")
                    except Exception as e:
                        print(f"[FACT BUILD] ❌ Failed to compute '{new_col}': {e}")

        # Collect all required fields
        required_fields = []
        for field in config["fields"]:
            if isinstance(field, str):
                required_fields.append(field)
            elif isinstance(field, dict):
                required_fields.extend(field.keys())

        # Verify all required fields exist
        missing = [col for col in required_fields if col not in fact_df.columns]
        if missing:
            print(f"[FACT LOAD] ❌ Missing columns: {missing}")
            print("Available columns:", fact_df.columns.tolist())
            continue

        # Determine primary key columns (combination of surrogate keys and date_ids)
        pk_columns = [col for col in required_fields if col.endswith('_sk') or col.endswith('date_id')]
        if not pk_columns:
            print(f"[FACT LOAD] ⚠️ No primary key columns found for {fact_name}")
            continue

        # Select final columns and remove duplicates based on PK columns
        fact_df = fact_df[required_fields].drop_duplicates(subset=pk_columns).reset_index(drop=True)

        try:
            # Create fact table with composite primary key
            create_stmt = f'CREATE TABLE IF NOT EXISTS "{schema}"."{fact_name}" ('
            
            # Add column definitions with proper types
            col_defs = []
            for col in fact_df.columns:
                if col.endswith('_sk') or col.endswith('date_id'):
                    sql_type = 'INTEGER'
                elif col in ['amount', 'loan_amount', 'interest', 'interest_rate']:
                    sql_type = 'NUMERIC'
                else:
                    sql_type = 'TEXT'
                col_defs.append(f'"{col}" {sql_type}')
            
            # Add primary key constraint - fixed version
            pk_columns_quoted = [f'"{col}"' for col in pk_columns]
            col_defs.append(f'PRIMARY KEY ({", ".join(pk_columns_quoted)})')
            
            create_stmt += ','.join(col_defs) + ');'
            
            with conn.cursor() as cur:
                cur.execute(create_stmt)
                conn.commit()
                print(f"[CREATE] 🧱 Created fact table '{schema}.{fact_name}' with PK: {pk_columns}")
            
            # Load data with upsert to handle duplicates
            upsert_dataframe_to_table(
                conn, 
                fact_df, 
                fact_name, 
                schema=schema,
                pk_cols=pk_columns  # Pass PK columns for proper upsert
            )
            
            print(f"[FACT LOAD] ✅ Loaded {len(fact_df)} rows into '{fact_name}'")
            fact_lookups[fact_name] = fact_df
            
        except Exception as e:
            print(f"[FACT LOAD] ❌ Load failed: {str(e)}")
            continue

    return fact_lookups

def enforce_olap_foreign_keys(conn, schema: str, olap_fk_config: Dict[str, List]):
    """
    Enforce foreign key relationships in OLAP schema
    """
    print("\n🔗 Enforcing OLAP foreign keys...")
    
    for table_name, fk_list in olap_fk_config.items():
        for fk_col, ref_table, ref_col in fk_list:
            constraint_name = f"fk_{table_name}_{fk_col}_to_{ref_table}"
            
            with conn.cursor() as cur:
                try:
                    # Drop existing constraint if it exists
                    drop_stmt = (
                        f'ALTER TABLE "{schema}"."{table_name}" '
                        f'DROP CONSTRAINT IF EXISTS "{constraint_name}"'
                    )
                    cur.execute(drop_stmt)
                    
                    # Create new foreign key constraint
                    add_stmt = (
                        f'ALTER TABLE "{schema}"."{table_name}" '
                        f'ADD CONSTRAINT "{constraint_name}" '
                        f'FOREIGN KEY ("{fk_col}") '
                        f'REFERENCES "{schema}"."{ref_table}" ("{ref_col}")'
                    )
                    cur.execute(add_stmt)
                    
                    print(f"✅ Added FK: {table_name}.{fk_col} → {ref_table}.{ref_col}")
                    
                except Exception as e:
                    print(f"❌ Failed to add FK {fk_col}: {str(e)}")
                    conn.rollback()
                    continue
    
    conn.commit()
    print("✅ Foreign key enforcement complete")

def generate_indexes_on_sk_and_date_ids(
    conn,
    tables: Dict[str, pd.DataFrame],
    schema: str = "public"
) -> None:
    """
    Creates indexes on *_sk and *_date_id columns in the given schema.
    Ignores tables that don't exist and logs index creation status.
    """
    created = []
    skipped = []

    with conn.cursor() as cur:
        for table_name, df in tables.items():
            # Check if table exists in DB
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                );
            """, (schema, table_name))
            exists = cur.fetchone()[0]

            if not exists:
                print(f"[INDEX] ⚠️ Skipping '{schema}.{table_name}' — table does not exist.")
                skipped.append(table_name)
                continue

            # Create indexes on *_sk and *_date_id columns
            for col in df.columns:
                if col.endswith("_sk") or col.endswith("_date_id"):
                    index_name = f"idx_{table_name}_{col}"
                    try:
                        cur.execute(
                            f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{schema}"."{table_name}"("{col}");'
                        )
                        created.append(index_name)
                    except Exception as e:
                        print(f"[INDEX] ❌ Failed: {index_name} — {e}")

    conn.commit()

    print(f"\n[INDEX] ✅ Created {len(created)} indexes:")
    for idx in created:
        print(f"   - {idx}")

    if skipped:
        print(f"\n[INDEX] ⚠️ Skipped {len(skipped)} tables (not found):")
        for tbl in skipped:
            print(f"   - {schema}.{tbl}")

def display_olap_info(conn, cfg: dict):
    """
    Display a simple overview of OLAP schema structure and relationships.
    
    Args:
        conn: Database connection
        cfg: Configuration dictionary containing OLAP schema
    """
    olap_schema = cfg["olap"]["schema"]
    print(f"\n=== OLAP Schema: {olap_schema} ===")

    # Get table information from database
    with conn.cursor() as cur:
        # 1. Display Dimensions
        print("\n📊 DIMENSION TABLES:")
        for dim_name in cfg["olap"]["dimensions"].keys():
            cur.execute(f"""
                SELECT COUNT(*) FROM {olap_schema}.{dim_name}
            """)
            count = cur.fetchone()[0]
            print(f"- {dim_name}: {count:,} rows")
            
        # 2. Display Facts
        print("\n📈 FACT TABLES:")
        for fact_name in cfg["olap"]["facts"].keys():
            cur.execute(f"""
                SELECT COUNT(*) FROM {olap_schema}.{fact_name}
            """)
            count = cur.fetchone()[0]
            print(f"- {fact_name}: {count:,} rows")
        
        # 3. Display Foreign Keys
        print("\n🔗 RELATIONSHIPS:")
        cur.execute("""
            SELECT 
                tc.table_name, kcu.column_name,
                ccu.table_name AS ref_table,
                ccu.column_name AS ref_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s
            ORDER BY tc.table_name;
        """, (olap_schema,))
        
        for row in cur.fetchall():
            print(f"- {row[0]}.{row[1]} → {row[2]}.{row[3]}")

# ================================ERD Generation===================================
def generate_erd_graph(
    fk_dict: dict, 
    schema_type: str = "both",  # "oltp", "olap", or "both"
    sk_dict: dict = None, 
    title: str = "Entity Relationship Diagram"
) -> Digraph:
    """
    Generate an ERD-style graph using Graphviz for OLTP and/or OLAP schemas.
    
    Args:
        fk_dict (dict): Foreign key relationships
        schema_type (str): Type of schema to display ("oltp", "olap", or "both")
        sk_dict (dict, optional): Surrogate key mappings
        title (str): Title of the diagram
    """
    dot = Digraph('erd')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='rounded')
    
    # Track tables to avoid duplicates
    added_tables = set()
    
    def is_olap_table(table: str) -> bool:
        return table.startswith(('fact_', 'dim_'))
    
    # Add nodes for all tables involved in relationships
    for src_table, fks in fk_dict.items():
        # Skip tables not in requested schema type
        if schema_type != "both":
            if schema_type == "oltp" and is_olap_table(src_table):
                continue
            if schema_type == "olap" and not is_olap_table(src_table):
                continue
                
        if src_table not in added_tables:
            dot.node(src_table, src_table)
            added_tables.add(src_table)
        
        for fk_col, ref_table, ref_col in fks:
            # Skip relationships not in requested schema type
            if schema_type != "both":
                if schema_type == "oltp" and is_olap_table(ref_table):
                    continue
                if schema_type == "olap" and not is_olap_table(ref_table):
                    continue
                    
            if ref_table not in added_tables:
                dot.node(ref_table, ref_table)
                added_tables.add(ref_table)
            
            dot.edge(src_table, ref_table, label=f"{fk_col}")
    
    # Add surrogate key relationships if provided and if not OLTP-only
    if sk_dict and schema_type != "oltp":
        for table, sks in sk_dict.items():
            if table not in added_tables:
                dot.node(table, table)
                added_tables.add(table)
            
            for sk in sks:
                dot.edge(table, table, sk, style='dashed')
    
    return dot

def generate_zulo_erd() -> Digraph:
    """
    Generate an ERD for the ZULO banking system showing relationships between OLTP and OLAP tables.
    """
    # Create a new directed graph
    dot = Digraph('ZULO Banking System ERD')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='rounded')

    # Define table columns for better visualization
    table_columns = {
        "oltp.transaction": [
            "transaction_id", "transaction_type", "amount", 
            "transaction_date", "transaction_date_id", "transaction_sk"
        ],
        "oltp.loan": [
            "loan_id", "loan_amount", "loan_type", "start_date", 
            "start_date_id", "end_date", "end_date_id", "interest_rate", "loan_sk"
        ],
        "oltp.account": [
            "account_id", "account_type", "balance", 
            "opening_date", "opening_date_id", "account_sk"
        ],
        "oltp.customer": [
            "customer_id", "first_name", "last_name", 
            "email", "phone", "customer_sk"
        ],
        "oltp.zulo_lookup": [
            "customer_sk", "account_sk", 
            "transaction_sk", "loan_sk"
        ],
        "olap.date_dim": [
            "date", "date_id", "year", "month", 
            "month_name", "quarter", "day", "day_of_week", 
            "is_weekend", "is_month_end"
        ],
        "olap.transaction_dim": [
            "transaction_sk", "transaction_type"
        ],
        "olap.account_dim": [
            "account_sk", "account_type"
        ],
        "olap.customer_dim": [
            "customer_sk", "first_name", "last_name", 
            "email", "phone"
        ],
        "olap.loan_dim": [
            "loan_sk", "loan_type"
        ],
        "olap.loan_fact": [
            "loan_sk", "customer_sk", "start_date_id", 
            "end_date_id", "loan_amount", "interest"
        ],
        "olap.transaction_fact": [
            "transaction_sk", "account_sk", "opening_date_id", 
            "transaction_date_id", "amount", "balance"
        ]
    }

    # Add nodes for each table with their columns
    for table, columns in table_columns.items():
        # Create a table-like structure for the node label
        label = f"<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0'>"
        label += f"<TR><TD COLSPAN='2' BGCOLOR='lightblue'><B>{table}</B></TD></TR>"
        for col in columns:
            label += f"<TR><TD ALIGN='LEFT'>{col}</TD></TR>"
        label += "</TABLE>>"
        
        # Add the node to the graph with schema-specific styling
        if table.startswith("oltp."):
            dot.node(table, label, color='blue')
        else:
            dot.node(table, label, color='green')

    # Define and add relationships
    relationships = [
        # OLTP relationships
        ("oltp.zulo_lookup", "oltp.customer", "customer_sk"),
        ("oltp.zulo_lookup", "oltp.loan", "loan_sk"),
        ("oltp.zulo_lookup", "oltp.account", "account_sk"),
        ("oltp.zulo_lookup", "oltp.transaction", "transaction_sk"),
        
        # OLAP relationships
        ("olap.customer_dim", "oltp.zulo_lookup", "customer_sk"),
        ("olap.loan_dim", "oltp.zulo_lookup", "loan_sk"),
        ("olap.account_dim", "oltp.zulo_lookup", "account_sk"),
        ("olap.transaction_dim", "oltp.zulo_lookup", "transaction_sk"),
        
        # Fact table relationships
        ("olap.loan_fact", "olap.customer_dim", "customer_sk"),
        ("olap.loan_fact", "olap.loan_dim", "loan_sk"),
        ("olap.loan_fact", "olap.date_dim", "start_date_id"),
        ("olap.loan_fact", "olap.date_dim", "end_date_id"),
        ("olap.transaction_fact", "olap.customer_dim", "customer_sk"),
        ("olap.transaction_fact", "olap.account_dim", "account_sk"),
        ("olap.transaction_fact", "olap.transaction_dim", "transaction_sk"),
        ("olap.transaction_fact", "olap.date_dim", "transaction_date_id"),
        ("olap.transaction_fact", "olap.date_dim", "opening_date_id")
    ]

    # Add edges for each relationship with schema-specific styling
    for from_table, to_table, key in relationships:
        if from_table.startswith("oltp.") and to_table.startswith("oltp."):
            dot.edge(from_table, to_table, label=key, color='blue')
        elif from_table.startswith("olap.") and to_table.startswith("olap."):
            dot.edge(from_table, to_table, label=key, color='green')
        else:
            dot.edge(from_table, to_table, label=key, color='red', style='dashed')

    return dot

# ================================ Validation & QA===================================
#  this step ensures facts and dimensions are not just created — but reliable and ready for BI tools, dashboards, or data science.
def qa_runner_with_pass(
    oltp_tables: Dict[str, pd.DataFrame],
    dim_lookups: Dict[str, pd.DataFrame],
    fact_tables: Dict[str, pd.DataFrame],
    checks: Dict[str, Dict[str, str]],
    verbose: bool = True
) -> bool:
    """
    Runs QA on facts and returns True if all checks pass, else False.

    Parameters:
    - oltp_tables: OLTP dataframes
    - dim_lookups: dimension lookup tables with surrogate keys
    - fact_tables: OLAP fact tables
    - checks: QA checks per fact table
    - verbose: if True, print check logs

    Returns:
    - qa_passed (bool): True if all checks pass
    """
    qa_passed = True
    failed_checks = []

    for fact_name, rules in checks.items():
        fact_df = fact_tables.get(fact_name)
        if fact_df is None:
            if verbose: print(f"❌ Fact table '{fact_name}' not found.")
            qa_passed = False
            failed_checks.append(f"{fact_name}: not found")
            continue

        # Row count check
        source_name = fact_name.replace("fact_", "")
        source_df = oltp_tables.get(source_name)
        if source_df is not None and verbose:
            print(f"📊 {fact_name}: OLTP = {len(source_df)} | OLAP = {len(fact_df)}")

        # Null checks
        for col in rules.get("check_not_null", []):
            nulls = fact_df[col].isna().sum()
            if nulls > 0:
                qa_passed = False
                failed_checks.append(f"{fact_name}: {nulls} NULLs in '{col}'")
                if verbose:
                    print(f"   ❌ NULL check failed: '{col}' has {nulls} missing")
            elif verbose:
                print(f"   ✅ '{col}' OK")

        # Orphan FK checks
        for sk_col, dim_name in rules.get("fk_checks", {}).items():
            dim_df = dim_lookups.get(dim_name)
            if dim_df is None:
                qa_passed = False
                failed_checks.append(f"{fact_name}: dimension '{dim_name}' missing")
                if verbose:
                    print(f"   ❌ Dimension '{dim_name}' not found.")
                continue

            valid_sks = dim_df[dim_df.columns[-1]]
            orphans = ~fact_df[sk_col].isin(valid_sks)
            orphan_count = orphans.sum()
            if orphan_count > 0:
                qa_passed = False
                failed_checks.append(f"{fact_name}: {orphan_count} orphan '{sk_col}'")
                if verbose:
                    print(f"   ❌ FK check failed: {orphan_count} '{sk_col}' not in '{dim_name}'")
            elif verbose:
                print(f"   ✅ FK '{sk_col}' linked to '{dim_name}'")

    if verbose:
        print("\n📋 QA Summary")
        print("✅ Passed" if qa_passed else f"❌ Failed: {len(failed_checks)} issues")
        for issue in failed_checks:
            print("   -", issue)

    return qa_passed

# ================================(Optional) Create Views============================
# ---View Builder for Fact Summaries
def create_materialized_fact_summary(
    conn,
    fact_table: str,
    dim_date: str,
    date_fk: str,
    group_fields: List[str],
    measures: Dict[str, str],
    schema: str = "public"
) -> None:
    """
    Creates a materialized view with grouped summaries for a fact table.
    """
    view_name = f"mv_{fact_table}_summary"
    full_view = f'"{schema}"."{view_name}"'
    full_fact = f'"{schema}"."{fact_table}"'
    full_date = f'"{schema}"."{dim_date}"'

    group_clause = ", ".join(group_fields)
    select_clause = ",\n    ".join(group_fields + [f"{agg}({col}) AS {agg}_{col}" for col, agg in measures.items()])

    stmt = f"""
    DROP MATERIALIZED VIEW IF EXISTS {full_view};
    CREATE MATERIALIZED VIEW {full_view} AS
    SELECT
        {select_clause}
    FROM {full_fact} f
    JOIN {full_date} d ON f."{date_fk}" = d.date_sk
    GROUP BY {group_clause};
    """
    try:
        cur = conn.cursor()
        cur.execute(stmt)
        conn.commit()
        cur.close()
        print(f"✅ Created materialized view: {view_name}")
    except Exception as e:
        print(f"❌ Failed to create materialized view {view_name}: {e}")

#================================12. DATA EXPORT =============================================
def export_sql_script(
    schema: str,
    tables: Dict[str, pd.DataFrame],
    foreign_keys: Dict[str, List[Tuple[str, str, str]]],
    output_sql_path: str
) -> None:
    """
    Generate SQL INSERT statements and save to a file.
    """
    sorted_table_names = topological_sort_tables(tables, foreign_keys)
    os.makedirs(os.path.dirname(output_sql_path), exist_ok=True)

    with open(output_sql_path, "w", encoding="utf-8") as f:
        for table in sorted_table_names:
            df = tables[table]
            columns = list(df.columns)
            col_str = ', '.join([f'"{col}"' for col in columns])
            for _, row in df.iterrows():
                values = []
                for val in row:
                    if pd.notnull(val):
                        escaped_val = str(val).replace("'", "''")
                        values.append(f"'{escaped_val}'")
                    else:
                        values.append('NULL')
                val_str = ', '.join(values)
                insert_stmt = f'INSERT INTO "{schema}"."{table}" ({col_str}) VALUES ({val_str});\n'
                f.write(insert_stmt)
    print(f"📝 SQL script saved to {output_sql_path}")

def save_tables_to_csv(tables: Dict[str, pd.DataFrame], export_dir: str) -> Dict[str, str]:
    """
    Save DataFrames to CSV files in a specified directory.

    Parameters:
    - tables: dictionary of table name -> DataFrame
    - export_dir: destination directory path

    Returns:
    - Dictionary of table name -> full CSV path
    """
    os.makedirs(export_dir, exist_ok=True)
    csv_paths = {}
    for name, df in tables.items():
        csv_path = os.path.join(export_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        csv_paths[name] = csv_path
        print(f"[save_tables_to_csv] Saved {name} to {csv_path}")
    return csv_paths

## ================================11. Automate the Pipeline===================================

'''
# df = read_data("data.csv")  # Local file
# df = read_data("1DdmNsrdBRLfzBdgtvzvFHZ7ejFpLlpwW", "gdrive")  # Google Drive
# df = read_data("https://example.com/data.csv")  # URL (auto-detected)
# df.head()
'''

