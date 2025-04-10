
import pandas as pd
import re
from typing import Tuple, List, Optional, Dict
import os
import sqlite3
from sqlalchemy import create_engine
import psycopg2
from dotenv import load_dotenv
from collections import defaultdict, deque


load_dotenv() # Load environment variables from .env file

# --- Data Extraction Function ---

def read_data(source, source_type='auto'):
    """
    Reads data from a CSV file, Google Drive, or a URL.
    
    Parameters:
    -----------
    source : str
        - File path (e.g., 'data.csv')
        - Google Drive file ID (e.g., '1DdmNsrdBRLfzBdgtvzvFHZ7ejFpLlpwW')
        - URL (e.g., 'https://example.com/data.csv')
    
    source_type : str, optional (default='auto')
        - 'file': Treat source as a local file path
        - 'gdrive': Treat source as a Google Drive file ID
        - 'url': Treat source as a direct downloadable URL
        - 'auto': Automatically detect the source type
    
    Returns:
    --------
    pd.DataFrame
        The loaded DataFrame.
    """
    if source_type == 'auto':
        if source.startswith(('http://', 'https://')):
            source_type = 'url'
        elif '/' in source or '\\' in source or source.endswith('.csv'):
            source_type = 'file'
        else:
            source_type = 'gdrive'
    
    if source_type == 'file':
        return pd.read_csv(source)
    elif source_type == 'gdrive':
        gdrive_url = f"https://drive.google.com/uc?id={source}"
        return pd.read_csv(gdrive_url)
    elif source_type == 'url':
        return pd.read_csv(source)
    else:
        raise ValueError("Invalid source_type. Use 'file', 'gdrive', 'url', or 'auto'.")


# --- Data Transformation functions ---

# 1. Handle Missing Values in Critical Columns
def handle_missing_critical(df: pd.DataFrame, critical_cols: List[str]) -> pd.DataFrame:
    initial_len = len(df)
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    print(f"[handle_missing_critical] Dropped {initial_len - len(df)} rows with nulls in {critical_cols}")
    return df

# 2. Ensure Correct Data Types (including dates)
def ensure_correct_dtypes(df: pd.DataFrame, datetime_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                print(f"[ensure_correct_dtypes] Converted '{col}' to datetime using day-first format.")
    return df


# 3. Remove Duplicates Based on Supposed PKs
def remove_duplicates_by_keys(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[col for col in key_cols if col in df.columns])
    print(f"[remove_duplicates_by_keys] Removed {before - len(df)} duplicate rows based on {key_cols}")
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
    if col in df.columns:
        split_data = df[col].str.strip().str.split(pat=' ', n=1, expand=True)
        for i, new_col in enumerate(new_cols):
            df[new_col] = split_data[i] if i < split_data.shape[1] else ''
        print(f"[split_compound_column] Split '{col}' into {new_cols}")
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


# 9. Define lowercase function
def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    print("[lowercase_columns] Column names converted to lowercase.")
    return df

## --- Data Modelling functions ---

def split_normalized_tables(df: pd.DataFrame, table_specs: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Splits a DataFrame into multiple normalized tables based on user-defined column lists.

    Parameters:
    - df: The cleaned DataFrame
    - table_specs: A dictionary where keys are table names and values are lists of column names for that table

    Returns:
    - A dictionary of DataFrames, each representing a normalized table
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    print("[split_normalized_tables] Column names normalized to lowercase.")

    split_tables = {}
    for table_name, columns in table_specs.items():
        cols = [col for col in columns if col in df.columns]
        if not cols:
            print(f"[split_normalized_tables] Warning: No matching columns found for table '{table_name}'")
            continue
        split_tables[table_name] = df[cols].drop_duplicates().reset_index(drop=True)
        print(f"[split_normalized_tables] Created table '{table_name}' with {len(split_tables[table_name])} rows.")

    return split_tables

def add_surrogate_key_to_table(
    tables: Dict[str, pd.DataFrame], 
    target_table: str, 
    key_name: str = "id"
) -> Dict[str, pd.DataFrame]:
    """
    Add a surrogate primary key to a specified table within a dictionary of DataFrames.

    Parameters:
    - tables: Dictionary of table_name -> DataFrame
    - target_table: Name of the table to modify
    - key_name: Name of the new surrogate key column

    Returns:
    - Updated dictionary with the surrogate key added to the target table
    """
    updated_tables = tables.copy()
    if target_table in updated_tables:
        df = updated_tables[target_table].copy()
        df.insert(0, key_name, range(1, len(df) + 1))
        updated_tables[target_table] = df
        print(f"[add_surrogate_key_to_table] ✅ Added '{key_name}' as surrogate key to '{target_table}'.")
    else:
        print(f"[add_surrogate_key_to_table] ❌ Table '{target_table}' not found.")
    return updated_tables



# ---- CSV Export Function ----
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


# --- Data Loading functions (postgreSQL) ---

def get_db_connection():
    """
    Create a PostgreSQL database connection using psycopg2.

    Returns:
    - PostgreSQL connection object, or None if connection fails
    """
    try:
        conn = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME_1")
        )
        print("[get_db_connection] Connection to PostgreSQL successful.")
        return conn
    except Exception as e:
        print(f"[get_db_connection] Connection failed: {e}")
        return None

def infer_keys_from_normalized_tables(
    tables: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, str, str]]]]:
    """
    Infer primary keys and foreign keys from normalized DataFrame tables.

    Assumptions:
    - Primary Key:
        ➤ Column ends with '_id'
        ➤ Column name contains the table name (singular form)
        ➤ Column contains unique values in its table
    - Foreign Key:
        ➤ Column ends with '_id'
        ➤ Column name matches a PK column in another table

    Returns:
    - primary_keys: {table_name: primary_key_column}
    - foreign_keys: {table_name: [(fk_column, referenced_table, referenced_column), ...]}
    """
    primary_keys = {}
    foreign_keys = {}
    table_name_roots = {
        tbl: tbl.lower().replace("_df", "").rstrip("s") for tbl in tables.keys()
    }

    print("\n[infer_keys_from_normalized_tables] 🔍 Starting key inference...\n")

    # Step 1: Detect Primary Keys
    for table_name, df in tables.items():
        root = table_name_roots[table_name]
        pk_candidates = [
            col for col in df.columns
            if col.endswith("_id") and root in col.lower() and df[col].is_unique
        ]
        if len(pk_candidates) == 1:
            primary_keys[table_name] = pk_candidates[0]
            print(f"✅ PRIMARY KEY for '{table_name}': {pk_candidates[0]}")
        elif len(pk_candidates) > 1:
            print(f"⚠️ Multiple PK candidates in '{table_name}': {pk_candidates} (no PK selected)")
        else:
            print(f"❌ No PK found in '{table_name}'")

    # Step 2: Detect Foreign Keys
    pk_lookup = {pk: tbl for tbl, pk in primary_keys.items()}

    for table_name, df in tables.items():
        fk_list = []
        for col in df.columns:
            if (
                col.endswith("_id") and
                col in pk_lookup and
                pk_lookup[col] != table_name
            ):
                fk_list.append((col, pk_lookup[col], col))
        if fk_list:
            foreign_keys[table_name] = fk_list
            for col, ref_table, ref_col in fk_list:
                print(f"🔗 FOREIGN KEY in '{table_name}': {col} → {ref_table}.{ref_col}")
        elif table_name not in primary_keys:
            print(f"ℹ️ '{table_name}' has no PK and no FK.")
        else:
            print(f"ℹ️ '{table_name}' has a PK but no FKs.")

    print(f"\n✅ Inference Summary: {len(primary_keys)} PKs and {sum(len(fks) for fks in foreign_keys.values())} FKs across {len(foreign_keys)} tables.\n")
    return primary_keys, foreign_keys



def topological_sort_tables(tables: Dict[str, pd.DataFrame], foreign_keys: Dict[str, List[Tuple[str, str, str]]]) -> List[str]:
    """
    Perform a topological sort of tables based on foreign key dependencies.
    Tables with no dependencies come first. Cycles are not handled here (assumes acyclic FK graph).
    """
    graph = defaultdict(set)
    in_degree = defaultdict(int)

    # Initialize all tables
    for table in tables:
        graph[table] = set()
        in_degree[table] = 0

    # Build dependency graph
    for table, fks in foreign_keys.items():
        for _, ref_table, _ in fks:
            graph[ref_table].add(table)
            in_degree[table] += 1

    # Kahn's algorithm
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
        print("⚠️ Cycle detected or missing dependency resolution.")
    return sorted_tables


def create_and_execute_schema_and_tables(
    conn,
    schema: str,
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, str],
    foreign_keys: Dict[str, List[Tuple[str, str, str]]]
) -> None:
    """
    Create and execute SQL statements to:
    - Drop schema if it exists (CASCADE)
    - Create schema
    - Drop and create tables in the schema with PKs and FKs, sorted by FK dependency
    - Handle failures with rollback
    - Report number of successful and failed table creations
    """
    if conn is None:
        print("[create_and_execute_schema_and_tables] ❌ No valid database connection.")
        return

    cur = conn.cursor()
    type_mapping = {
        'object': 'TEXT',
        'int64': 'INTEGER',
        'float64': 'FLOAT',
        'datetime64[ns]': 'TIMESTAMP',
        'bool': 'BOOLEAN'
    }

    successful_tables = 0
    failed_tables = 0

    try:
        # Step 1: Drop and create schema
        print(f"[create_and_execute_schema_and_tables] ❌ Dropping schema '{schema}' if it exists...")
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
        conn.commit()

        print(f"[create_and_execute_schema_and_tables] ✅ Creating schema '{schema}'...")
        cur.execute(f'CREATE SCHEMA "{schema}";')
        conn.commit()

        # Step 2: Sort tables by FK dependencies
        sorted_table_names = topological_sort_tables(tables, foreign_keys)

        # Step 3: Create tables in sorted order
        for table_name in sorted_table_names:
            df = tables[table_name]
            print(f"[create_and_execute_schema_and_tables] ❌ Dropping table '{table_name}' if it exists...")
            cur.execute(f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE;')
            conn.commit()

            column_defs = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                sql_type = type_mapping.get(dtype, 'TEXT')
                column_defs.append(f'"{col}" {sql_type}')

            # Add primary key
            if table_name in primary_keys:
                pk = primary_keys[table_name]
                column_defs.append(f'PRIMARY KEY ("{pk}")')

            # Add foreign keys
            if table_name in foreign_keys:
                for fk_col, ref_table, ref_col in foreign_keys[table_name]:
                    fk = f'FOREIGN KEY ("{fk_col}") REFERENCES "{schema}"."{ref_table}"("{ref_col}")'
                    column_defs.append(fk)

            create_stmt = f'CREATE TABLE "{schema}"."{table_name}" (\n    ' + ",\n    ".join(column_defs) + "\n);"
            print(f"[create_and_execute_schema_and_tables] ✅ Creating table '{table_name}'...")
            try:
                cur.execute(create_stmt)
                conn.commit()
                successful_tables += 1
            except Exception as table_err:
                print(f"❌ Failed to create table '{table_name}': {table_err}")
                conn.rollback()
                failed_tables += 1

    except Exception as err:
        print(f"❌ Schema creation failed: {err}")
        conn.rollback()

    finally:
        cur.close()
        print(f"\n[create_and_execute_schema_and_tables] ✅ Execution finished for schema '{schema}'.")
        print(f"📊 Summary: {successful_tables} tables created, {failed_tables} failed.")


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


def load_db(
    conn,
    tables: Dict[str, pd.DataFrame],
    foreign_keys: Dict[str, List[Tuple[str, str, str]]],
    schema: Optional[str] = "public"
) -> None:
    """
    Load DataFrames into PostgreSQL using psycopg2 with FK-safe ordering.

    Parameters:
    - conn: psycopg2 connection object
    - tables: dictionary of table name -> DataFrame
    - foreign_keys: foreign key mapping (used to sort tables by dependency)
    - schema: target schema name (default: public)
    """
    if conn is None:
        print("❌ No valid database connection.")
        return

    cursor = conn.cursor()
    sorted_table_names = topological_sort_tables(tables, foreign_keys)

    for name in sorted_table_names:
        df = tables[name]
        print(f"🚚 Loading '{schema}.{name}'...")

        inserted = 0
        for _, row in df.iterrows():
            columns = list(row.index)
            col_str = ', '.join([f'"{col}"' for col in columns])
            values = []
            for val in row:
                if pd.notnull(val):
                    escaped_val = str(val).replace("'", "''")
                    values.append(f"'{escaped_val}'")
                else:
                    values.append('NULL')
            val_str = ', '.join(values)
            insert_stmt = f'INSERT INTO "{schema}"."{name}" ({col_str}) VALUES ({val_str});'
            try:
                cursor.execute(insert_stmt)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"❌ Error inserting into '{name}': {e}")
                continue

        conn.commit()
        print(f"✅ {inserted} rows loaded into '{schema}.{name}'.")

    cursor.close()
    print("\n📊 Load complete for all tables.")






