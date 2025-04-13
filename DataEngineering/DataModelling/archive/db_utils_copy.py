# ETL Pipeline: Extract, Transform, Load
# This script implements an ETL pipeline that extracts data from various sources, transforms it, and loads it into a PostgreSQL database.

import pandas as pd
import re
from typing import Tuple, List, Optional, Dict
import os
import sqlite3
from sqlalchemy import create_engine
import psycopg2
from dotenv import load_dotenv
from collections import defaultdict, deque
import holidays

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
    """
    Convert specified columns to datetime using flexible parsing to support multiple date formats.
    Silences warnings by avoiding conflicting parameters.
    """
    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)

                        # Try parsing with dayfirst
                        parsed_dayfirst = pd.to_datetime(df[col], errors='coerce', dayfirst=True, infer_datetime_format=True)
                        # Try parsing with yearfirst
                        parsed_yearfirst = pd.to_datetime(df[col], errors='coerce', dayfirst=False, yearfirst=True, infer_datetime_format=True)

                    # Use whichever parsing yields more valid entries
                    if parsed_dayfirst.notna().sum() >= parsed_yearfirst.notna().sum():
                        df[col] = parsed_dayfirst
                        print(f"[ensure_correct_dtypes] Converted '{col}' using day-first format.")
                    else:
                        df[col] = parsed_yearfirst
                        print(f"[ensure_correct_dtypes] Converted '{col}' using year-first format.")
                except Exception as e:
                    print(f"❌ Failed to convert '{col}' to datetime: {e}")
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

# 10. Standardize Column Names
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names in a single DataFrame to snake_case format (e.g., 'CustomerID' -> 'customer_id').

    Parameters:
    - df: Input DataFrame

    Returns:
    - DataFrame with standardized snake_case column names
    """
    import re

    def to_snake_case(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    df_copy = df.copy()
    df_copy.columns = [to_snake_case(col.strip()) for col in df_copy.columns]
    print("[standardize_columns_in_df] Renamed columns to snake_case.")
    return df_copy


# 11. Generate Date Dimension
def generate_date_dim(df: pd.DataFrame, date_columns: List[str], id_col: str = 'date_id', country: str = 'US') -> pd.DataFrame:
    """
    Generate a date dimension DataFrame from one or more date columns, including holiday flag.

    Parameters:
    - df: original DataFrame containing date columns
    - date_columns: list of date column names to extract unique dates from
    - id_col: name of the surrogate key column (default: 'date_id')
    - country: ISO country code for holiday lookup (default: 'US')

    Returns:
    - A DataFrame representing the date dimension
    """
    all_dates = pd.Series(dtype='datetime64[ns]')
    for col in date_columns:
        if col in df.columns:
            all_dates = pd.concat([all_dates, df[col].dropna()])

    unique_dates = pd.to_datetime(all_dates.dropna().unique())
    unique_dates = pd.Series(pd.to_datetime(sorted(unique_dates))).drop_duplicates()

    date_dim = pd.DataFrame({
        'full_date': unique_dates
    })
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
        country_holidays = holidays.country_holidays(country)
        date_dim['is_holiday'] = date_dim['full_date'].isin(country_holidays)
    except Exception as e:
        print(f"[generate_date_dim] ⚠️ Could not determine holidays for country '{country}': {e}")
        date_dim['is_holiday'] = False

    print(f"[generate_date_dim] ✅ Created date dimension with {len(date_dim)} unique dates (including holidays).")
    return date_dim






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
        table_df = df[cols].dropna().drop_duplicates().reset_index(drop=True)
        split_tables[table_name] = table_df
        print(f"[split_normalized_tables]  Created table '{table_name}' with {len(table_df)} non-null rows.")

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

def get_db_connection():# Data Loading functions (postgreSQL)
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
 
def get_db_connection(env_prefix: str = "DB_") -> psycopg2.extensions.connection:
    """
    Establish a PostgreSQL connection using environment variables with a specified prefix.

    Parameters:
    - env_prefix (str): Prefix for environment variable keys (e.g., 'DB_', 'STAGE_DB_', etc.)

    Required environment variables (with prefix):
    - {PREFIX}USER
    - {PREFIX}PASSWORD
    - {PREFIX}HOST
    - {PREFIX}PORT
    - {PREFIX}NAME

    Returns:
    - psycopg2 connection object or None if connection fails
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
        print(f"[get_db_connection] ✅ Connected to database '{database}' using prefix '{env_prefix}'")
        return conn
    except Exception as e:
        print(f"[get_db_connection] ❌ Failed to connect using prefix '{env_prefix}': {e}")
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


def infer_keys_extended(tables: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, str, str]]], Dict[str, List[str]], str]:
    """
    Extended key inference for OLTP and OLAP:
    - Infers Primary Keys (PK)
    - Infers Foreign Keys (FK)
    - Detects Surrogate Keys (SK)

    Returns:
    - primary_keys: {table_name: primary_key_column}
    - foreign_keys: {table_name: [(fk_column, referenced_table, referenced_column), ...]}
    - surrogate_keys: {table_name: [surrogate_key_columns]}
    - formatted_report: str formatted report as a summary
    """
    primary_keys = {}
    foreign_keys = {}
    surrogate_keys = {}
    report_lines = []

    table_name_roots = {
        tbl: tbl.lower().replace("_df", "").replace("dim_", "").replace("fact_", "").rstrip("s")
        for tbl in tables.keys()
    }

    report_lines.append("\n🔍 **Extended Key Inference Report**\n")

    # Step 1: Detect Primary Keys
    report_lines.append("📌 **Primary Keys**")
    for table_name, df in tables.items():
        root = table_name_roots[table_name]
        pk_candidates = [
            col for col in df.columns
            if col.endswith("_id") and root in col.lower() and df[col].is_unique
        ]
        if len(pk_candidates) == 1:
            primary_keys[table_name] = pk_candidates[0]
            report_lines.append(f"   ✔️ {table_name:<20} → {pk_candidates[0]}")
        elif len(pk_candidates) > 1:
            report_lines.append(f"   ⚠️ {table_name:<20} → Multiple PKs: {pk_candidates}")
        else:
            report_lines.append(f"   ❌ {table_name:<20} → No primary key detected")

    # Step 2: Detect Foreign Keys
    pk_lookup = {pk: tbl for tbl, pk in primary_keys.items()}

    report_lines.append("\n📎 **Foreign Keys**")
    for table_name, df in tables.items():
        fk_list = []
        for col in df.columns:
            if col.endswith("_id") and col in pk_lookup and pk_lookup[col] != table_name:
                fk_list.append((col, pk_lookup[col], col))
        if fk_list:
            foreign_keys[table_name] = fk_list
            for col, ref_table, ref_col in fk_list:
                report_lines.append(f"   🔗 {table_name:<20} → {col} → {ref_table}.{ref_col}")
        elif table_name not in primary_keys:
            report_lines.append(f"   ℹ️ {table_name:<20} → No PK or FK")
        else:
            report_lines.append(f"   ℹ️ {table_name:<20} → Has PK, no FKs")

    # Step 3: Detect Surrogate Keys
    report_lines.append("\n🧬 **Surrogate Key Candidates**")
    for table_name, df in tables.items():
        surrogate_keys[table_name] = [
            col for col in df.columns
            if col.endswith('_id') and pd.api.types.is_integer_dtype(df[col])
        ]
        if surrogate_keys[table_name]:
            report_lines.append(f"   🧬 {table_name:<20} → {', '.join(surrogate_keys[table_name])}")

    report_lines.append(f"\n✅ **Inference Summary**")
    report_lines.append(f"   ➤ Total Tables: {len(tables)}")
    report_lines.append(f"   ➤ Primary Keys: {len(primary_keys)}")
    report_lines.append(f"   ➤ Foreign Keys: {sum(len(v) for v in foreign_keys.values())}")
    report_lines.append(f"   ➤ Surrogate Keys: {len(surrogate_keys)} tables evaluated\n")

    formatted_report = "\n".join(report_lines)
    return primary_keys, foreign_keys, surrogate_keys, formatted_report


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
    foreign_keys: Dict[str, List[Tuple[str, str, str]]],
    surrogate_keys: Dict[str, List[str]]
) -> None:
    """
    Create and execute SQL statements to support OLTP and OLAP:
    - Drop schema if it exists (CASCADE)
    - Create schema
    - Drop and create tables in sorted FK dependency order
    - Add primary keys, foreign key constraints, and surrogate keys where needed
    - Handle errors with rollback
    - Print execution summary
    """
    if conn is None:
        print("[create_and_execute_schema_and_tables] ❌ No valid database connection.")
        return

    cur = conn.cursor()
    type_mapping = {
        'object': 'TEXT',
        'int64': 'INTEGER',
        'Int64': 'INTEGER',
        'float64': 'FLOAT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP'
    }

    successful_tables = 0
    failed_tables = 0

    try:
        print(f"[create_and_execute_schema_and_tables] 🚫 Dropping schema '{schema}' if it exists...")
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
        conn.commit()

        print(f"[create_and_execute_schema_and_tables] ✅ Creating schema '{schema}'...")
        cur.execute(f'CREATE SCHEMA "{schema}";')
        conn.commit()

        sorted_table_names = topological_sort_tables(tables, foreign_keys)

        for table_name in sorted_table_names:
            df = tables[table_name]
            print(f"[create_and_execute_schema_and_tables] 🚫 Dropping table '{table_name}' if it exists...")
            cur.execute(f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE;')
            conn.commit()

            column_defs = []

            # Inject surrogate key if no PK is defined
            if table_name not in primary_keys:
                surrogate_name = f"{table_name}_sk"
                column_defs.append(f'"{surrogate_name}" SERIAL PRIMARY KEY')
                print(f"[create_and_execute_schema_and_tables] 🧬 Added surrogate key '{surrogate_name}' to '{table_name}'")

            for col in df.columns:
                dtype = str(df[col].dtype)
                sql_type = type_mapping.get(dtype, 'TEXT')
                column_defs.append(f'"{col}" {sql_type}')

            # Add explicit PK if exists
            if table_name in primary_keys:
                pk = primary_keys[table_name]
                column_defs.append(f'PRIMARY KEY ("{pk}")')

            # Add foreign keys if exist
            if table_name in foreign_keys:
                for fk_col, ref_table, ref_col in foreign_keys[table_name]:
                    fk = f'FOREIGN KEY ("{fk_col}") REFERENCES "{schema}"."{ref_table}"("{ref_col}")'
                    column_defs.append(fk)

            create_stmt = f'CREATE TABLE "{schema}"."{table_name}" (\n    ' + ",\n    ".join(column_defs) + "\n);"

            print(f"[create_and_execute_schema_and_tables] 🛠️ Creating table '{table_name}'...")
            try:
                cur.execute(create_stmt)
                conn.commit()
                successful_tables += 1
                print(f"[create_and_execute_schema_and_tables] ✅ Successfully created '{table_name}'.")
            except Exception as table_err:
                print(f"[create_and_execute_schema_and_tables] ❌ Failed to create table '{table_name}': {table_err}")
                conn.rollback()
                failed_tables += 1

    except Exception as err:
        print(f"[create_and_execute_schema_and_tables] ❌ Schema creation failed: {err}")
        conn.rollback()

    finally:
        cur.close()
        print(f"\n[create_and_execute_schema_and_tables] ✅ Schema creation process complete for '{schema}'.")
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


import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict


# Step 1: Calculate interest paid on each loan (if loan_amount and interest_rate are present)
def compute_interest(df: pd.DataFrame, principal_col: str, rate_col: str, new_col: str = 'computed_interest') -> pd.DataFrame:
    """
    Compute simple interest and add as a new column.
    Assumes simple interest: Interest = Principal × Rate
    """
    if principal_col in df.columns and rate_col in df.columns:
        df[new_col] = (df[principal_col] * df[rate_col] / 100).round(2)
        print(f"[compute_interest] ✅ Computed '{new_col}' from {principal_col} * {rate_col} / 100")
    return df


# Helper to get date_id for each raw date column

def auto_map_date_ids(df: pd.DataFrame, date_dim: pd.DataFrame, date_key='date_id') -> pd.DataFrame:
    """
    Dynamically map all datetime columns in the DataFrame to their corresponding date_id from date_dim.
    Enforces integer type for all *_date_id columns.

    Parameters:
    - df: original DataFrame (with datetime64[ns] columns)
    - date_dim: date dimension DataFrame with `full_date` and `date_id`
    - date_key: the name of the key in date_dim to map back to (default: 'date_id')

    Returns:
    - df with new *_id columns next to each datetime column
    """
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    result_df = df.copy()

    for col in datetime_cols:
        date_map = date_dim[['full_date', date_key]].drop_duplicates()
        result_df = result_df.merge(date_map, how='left', left_on=col, right_on='full_date')
        new_col = f"{col.lower()}_id"
        result_df = result_df.drop(columns=['full_date']).rename(columns={date_key: new_col})
        result_df[new_col] = result_df[new_col].astype('Int64')  # nullable integer type
        print(f"[auto_map_date_ids] ✅ Mapped '{col}' to '{new_col}' and cast to Int64.")

    return result_df

def get_normalized_oltp_tables(raw_df: pd.DataFrame, table_definitions: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    print("[get_normalized_oltp_tables] 🧪 Normalizing base OLTP tables...")
    return split_normalized_tables(raw_df, table_definitions)



def prepare_olap_foreign_keys(
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, str]
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Prepares foreign key mappings for OLAP models following best practices:
    - fact tables → dimension tables
    - dimension tables → normalized base tables
    - removes any fact → raw table relationships

    Parameters:
    - tables: all normalized + OLAP tables
    - primary_keys: dictionary of inferred primary keys

    Returns:
    - foreign_keys: dictionary of table_name -> list of (fk_column, referenced_table, referenced_column)
    """
    foreign_keys = {}

    pk_lookup = {col: tbl for tbl, col in primary_keys.items()}

    for table_name, df in tables.items():
        fk_list = []

        for col in df.columns:
            if not col.endswith('_id') or col not in pk_lookup:
                continue

            ref_table = pk_lookup[col]

            # fact → dim only
            if table_name.endswith('_fact') and ref_table.startswith('dim_'):
                fk_list.append((col, ref_table, col))

            # dim → raw only
            elif table_name.startswith('dim_') and not ref_table.startswith(('dim_', 'fact_')):
                fk_list.append((col, ref_table, col))

        if fk_list:
            foreign_keys[table_name] = fk_list

    print(f"[prepare_olap_foreign_keys] ✅ Prepared {len(foreign_keys)} FK mappings for OLAP model.")
    return foreign_keys




# etl_pipeline.py (revised for OLTP & OLAP separation)

import pandas as pd
from typing import Dict, List, Tuple, Optional
from db_utils import *
from config import *


def clean_data(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Dynamic data cleaning based on config"""
    df = standardize_columns(raw_df)
    
    # Handle missing values
    if 'critical_columns' in config['input']:
        df = handle_missing_critical(df, config['input']['critical_columns'])
    
    # Convert date columns
    if 'date_columns' in config['input']:
        df = ensure_correct_dtypes(df, config['input']['date_columns'])
    
    # Split compound columns
    if 'split_columns' in config['cleaning']:
        for source_col, new_cols in config['cleaning']['split_columns'].items():
            df = split_compound_column(df, source_col, new_cols)
    
    return df

def read_data1(source, source_type='auto'):
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

def generate_index_pks1(tables: dict, pk: dict = None) -> dict:
    """
    Adds synthetic primary keys to tables that lack one,
    based on the rule:
        table_name == column name (minus _id), and column is unique.

    Parameters:
    - tables: dict of table_name -> DataFrame
    - pk: optional dict (optional)

    Returns:
    - pk: updated dict of table -> PK columns
    """
    pk = pk or {}

    for table_name, df in tables.items():
        # Already has a defined PK
        if table_name in pk and pk[table_name]:
            continue

        # Try to match table name (singular) + _id
        singular_name = table_name.rstrip("s")  # naive singularization
        expected_col = f"{singular_name}_id"

        if expected_col in df.columns and df[expected_col].is_unique:
            pk[table_name] = [expected_col]
            print(f"✅ Inferred PK for '{table_name}': {expected_col}")
            continue

        # If nothing matches, create synthetic PK
        synthetic_pk = f"{table_name}_id"
        print(f"⚠️ No valid PK found for '{table_name}'. Generating synthetic: '{synthetic_pk}'")

        df = df.reset_index(drop=True)
        df[synthetic_pk] = range(1, len(df) + 1)
        df = df[[synthetic_pk] + [c for c in df.columns if c != synthetic_pk]]
        tables[table_name] = df
        pk[table_name] = [synthetic_pk]

    return pk

# === STEP 1: OLTP WORKFLOW ===

def run_oltp_pipeline(conn, tables: Dict[str, pd.DataFrame], schema: str = "oltp"):
    print("\n🔧 Running OLTP pipeline...")
    primary_keys, foreign_keys, surrogate_keys, _ = infer_keys_extended(tables)
    create_schema_and_tables(conn, schema, tables, primary_keys, foreign_keys, surrogate_keys)
    load_to_schema(conn, tables, foreign_keys, schema)
    return primary_keys, foreign_keys, surrogate_keys


# === STEP 2: OLAP WORKFLOW ===

def run_olap_pipeline(conn, tables: Dict[str, pd.DataFrame], pk: Dict[str, str], schema: str = "olap"):
    print("\n📊 Running OLAP pipeline...")
    # Build FK relationships compliant with OLAP modeling
    fk_olap = prepare_olap_foreign_keys(tables, pk)
    # Reuse surrogate keys or add more if needed
    _, _, sk_olap, _ = infer_keys_extended(tables)
    create_schema_and_tables(conn, schema, tables, pk, fk_olap, sk_olap)
    load_to_schema(conn, tables, fk_olap, schema)


# === MAIN DRIVER ===

def main():
    print("🔄 Starting ETL pipeline...")
    conn = get_db_connection(env_prefix="DB_")
    if not conn:
        return

    # --- 1. Load and normalize OLTP tables ---
    oltp_tables = get_normalized_oltp_tables()
    pk_oltp, fk_oltp, sk_oltp = run_oltp_pipeline(conn, oltp_tables)

    # --- 2. Derive OLAP tables ---
    olap_tables = get_olap_tables_from_oltp(oltp_tables)
    run_olap_pipeline(conn, olap_tables, pk_oltp)

    conn.close()
    print("\n✅ Full OLTP & OLAP ETL pipeline completed.")


if __name__ == "__main__":
    main()


# === 6. FLEXIBLE DATA READER ===
def read_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif ext == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    




def extract_dim_tables(
    normalized_tables: Dict[str, pd.DataFrame],
    dimension_defs: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    dim_tables = {}
    for dim_name, dim_columns in dimension_defs.items():
        for source_name, source_df in normalized_tables.items():
            if all(col in source_df.columns for col in dim_columns):
                dim_df = source_df[dim_columns].drop_duplicates().reset_index(drop=True)
                dim_tables[dim_name] = dim_df
                break
    return dim_tables


def get_olap_tables_from_oltp(oltp_tables: Dict[str, pd.DataFrame], dimension_defs: Dict[str, List[str]], fact_defs: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    print("[get_olap_tables_from_oltp] 🧠 Transforming OLTP to OLAP (dim/fact)...")
    dim_tables = extract_dim_tables(oltp_tables, dimension_defs)
    fact_tables = {}
    for fact_name, fact_columns in fact_defs.items():
        for source_name, source_df in oltp_tables.items():
            if all(col in source_df.columns for col in fact_columns):
                fact_df = source_df[fact_columns].drop_duplicates().reset_index(drop=True)
                fact_tables[fact_name] = fact_df
                break
    return {
        **dim_tables,
        **fact_tables
    }


def infer_keys_from_normalized_tables(
    tables: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, str, str]]]]:
    """
    Infer primary and foreign keys from normalized DataFrame tables based on naming and uniqueness.

    Assumptions:
    - Primary Key:
        - Column ends with '_id'
        - Column name contains the singular form of the table name
        - Column contains unique values within its table
    - Foreign Key:
        - Column ends with '_id'
        - Column name matches the PK column of another table

    Parameters:
    - tables (dict): Mapping of table_name -> DataFrame

    Returns:
    - primary_keys (dict): {table_name: primary_key_column}
    - foreign_keys (dict): {table_name: [(fk_column, referenced_table, referenced_column), ...]}
    """
    primary_keys = {}
    foreign_keys = {}

    # Create lookup for simplified root table names
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
            print(f"ℹ️ '{table_name}' has no PK and no FKs.")
        else:
            print(f"ℹ️ '{table_name}' has a PK but no FKs.")

    print(
        f"\n✅ Inference Summary: {len(primary_keys)} PKs and "
        f"{sum(len(fks) for fks in foreign_keys.values())} FKs across "
        f"{len(foreign_keys)} tables.\n"
    )

    return primary_keys, foreign_keys


def generate_index_pks(tables: dict, pk: dict) -> dict:
    """
    For tables without primary keys, add an index column and mark it as PK.
    """
    for table_name, df in tables.items():
        if table_name not in pk or not pk[table_name]:  # No PK defined
            index_col = f"{table_name}_id"
            print(f"⚠️ No primary key found for '{table_name}'. Generating '{index_col}'...")

            df = df.reset_index(drop=True)
            df[index_col] = range(1, len(df) + 1)

            # Place index column at the front
            cols = [index_col] + [c for c in df.columns if c != index_col]
            tables[table_name] = df[cols]

            # Register it as the PK
            pk[table_name] = [index_col]
    return pk

# === 5. OLAP FK INFERENCE ===
def prepare_olap_foreign_keys(tables: Dict[str, pd.DataFrame], primary_keys: Dict[str, str]) -> Dict[str, List[Tuple[str, str, str]]]:
    foreign_keys = {}
    pk_lookup = {col: tbl for tbl, col in primary_keys.items()}
    for table_name, df in tables.items():
        fk_list = []
        for col in df.columns:
            if not col.endswith('_id') or col not in pk_lookup:
                continue
            ref_table = pk_lookup[col]
            if table_name.endswith('_fact') and ref_table.startswith('dim_'):
                fk_list.append((col, ref_table, col))
            elif table_name.startswith('dim_') and not ref_table.startswith(('dim_', 'fact_')):
                fk_list.append((col, ref_table, col))
        if fk_list:
            foreign_keys[table_name] = fk_list
    print(f"[prepare_olap_foreign_keys] ✅ Prepared {len(foreign_keys)} FK mappings for OLAP model.")
    return foreign_keys

def split_normalized_tables1(df: pd.DataFrame, table_specs: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    print("[split_normalized_tables] Column names normalized to lowercase.")
    split_tables = {}
    for table_name, columns in table_specs.items():
        cols = [col for col in columns if col in df.columns]
        if not cols:
            print(f"[split_normalized_tables] Warning: No matching columns found for table '{table_name}'")
            continue
        table_df = df[cols].dropna().drop_duplicates().reset_index(drop=True)
        split_tables[table_name] = table_df
        print(f"[split_normalized_tables] ✅ Created table '{table_name}' with {len(table_df)} non-null rows.")
    return split_tables


def create_and_execute_schema_and_tables(
    conn,
    schema: str,
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, List[str]],
    foreign_keys: Dict[str, List[Tuple[str, str, str]]],
    surrogate_keys: Dict[str, List[str]] = None
) -> None:
    """
    Create and execute SQL schema and table creation for OLTP or OLAP models.

    - Drops and recreates schema
    - Drops and creates tables in FK order
    - Adds PKs, FKs, and SKs where defined
    """
    if conn is None:
        print("[❌] No valid database connection.")
        return

    surrogate_keys = surrogate_keys or {}
    cur = conn.cursor()

    type_mapping = {
        "object": "TEXT",
        "int64": "INTEGER", "Int64": "INTEGER",
        "float64": "FLOAT",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP"
    }

    def get_sql_type(dtype: str) -> str:
        return type_mapping.get(dtype, "TEXT")

    successful_tables, failed_tables = 0, 0

    try:
        print(f"🧹 Dropping schema '{schema}' if exists...")
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
        conn.commit()

        print(f"🧱 Creating schema '{schema}'...")
        cur.execute(f'CREATE SCHEMA "{schema}";')
        conn.commit()

        sorted_table_names = list(tables.keys())

        for table_name in sorted_table_names:
            df = tables[table_name]

            print(f"\n🔁 Creating table '{table_name}'...")
            cur.execute(f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE;')
            conn.commit()

            column_defs = []

            # Surrogate key (OLAP)
            surrogate_set = set(surrogate_keys.get(table_name, []))
            for i, sk in enumerate(surrogate_keys.get(table_name, [])):
                column_defs.append(f'"{sk}" SERIAL{" PRIMARY KEY" if i == 0 else ""}')
                print(f"🧬 Surrogate key added: {sk}")

            # Regular columns (exclude SKs)
            for col in df.columns:
                if col not in surrogate_set:
                    sql_type = get_sql_type(str(df[col].dtype))
                    column_defs.append(f'"{col}" {sql_type}')

            # Primary key (OLTP)
            if table_name in primary_keys:
                pk_cols = primary_keys[table_name]
                pk_def = 'PRIMARY KEY (' + ', '.join([f'"{col}"' for col in pk_cols]) + ')'
                column_defs.append(pk_def)

            # Foreign keys
            if table_name in foreign_keys:
                for fk_col, ref_table, ref_col in foreign_keys[table_name]:
                    column_defs.append(
                        f'FOREIGN KEY ("{fk_col}") REFERENCES "{schema}"."{ref_table}"("{ref_col}")'
                    )

            create_stmt = f'CREATE TABLE "{schema}"."{table_name}" (\n    ' + ",\n    ".join(column_defs) + "\n);"

            try:
                cur.execute(create_stmt)
                conn.commit()
                successful_tables += 1
                print(f"✅ Created: {table_name}")
            except Exception as table_err:
                conn.rollback()
                print(f"❌ Failed to create '{table_name}': {table_err}")
                failed_tables += 1

    except Exception as schema_err:
        conn.rollback()
        print(f"[❌] Schema creation failed: {schema_err}")
    finally:
        cur.close()
        print(f"\n📋 Schema '{schema}' creation summary:")
        print(f"✅ Tables created: {successful_tables}, ❌ Failed: {failed_tables}")
        print(f"🔗 Foreign keys added: {sum(len(v) for v in foreign_keys.values()) if foreign_keys else 0}")


def copy_dataframe_to_table(
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


def copy_dataframe_to_table1(
    conn,
    df: pd.DataFrame,
    table_name: str,
    schema: str = "public"
) -> None:
    """
    Uses PostgreSQL's COPY FROM STDIN to bulk load a DataFrame into a target table.

    Parameters:
    - conn: psycopg2 connection object
    - df: DataFrame to load
    - table_name: target table name
    - schema: target schema (default "public")
    """
    if df.empty:
        print(f"[COPY] ⚠️ Skipping '{schema}.{table_name}' (empty DataFrame)")
        return

    # Create CSV buffer
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t", na_rep="\\N")
    buffer.seek(0)

    cursor = conn.cursor()

    try:
        print(f"[COPY] 🚀 Copying {len(df)} rows into \"{schema}.{table_name}\"...")
        cursor.execute(f'SET search_path TO "{schema}";')
        cursor.copy_from(buffer, table_name, sep="\t", null="\\N", columns=list(df.columns))
        conn.commit()
        print(f"[COPY] ✅ Successfully loaded \"{schema}.{table_name}\"")
    except Exception as e:
        conn.rollback()
        print(f"[COPY] ❌ Failed to load \"{schema}.{table_name}\": {e}")
    finally:
        cursor.close()

def topological_sort_tables(tables: Dict[str, pd.DataFrame], foreign_keys: Dict[str, List[Tuple[str, str, str]]]) -> List[str]:
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    for table in tables:
        graph[table] = set()
        in_degree[table] = 0
    for table, fks in foreign_keys.items():
        for _, ref_table, _ in fks:
            graph[ref_table].add(table)
            in_degree[table] += 1
    queue = deque([t for t in tables if in_degree[t] == 0])
    sorted_tables = []
    while queue:
        node = queue.popleft()
        sorted_tables.append(node)
        for dependent in graph[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    return sorted_tables