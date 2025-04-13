# db_utils.py - consolidated and cleaned for OLTP & OLAP workflows

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import psycopg2
from collections import defaultdict, deque
import re
import os
import psycopg2
from dotenv import load_dotenv
import holidays
import warnings
from datetime import datetime, timedelta
from config import ETL_CONFIG
from IPython.display import display, HTML
import ipywidgets as widgets
import io
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


load_dotenv() # Load environment variables from .env file


# ==================================== DB CONNECTION =========================================

# Establish a PostgreSQL connection using environment variables
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
        print(f"[get_db_connection] ✅ Connected to '{conn.dsn}' using prefix '{env_prefix}'")
        
        return conn
    except Exception as e:
        print(f"[get_db_connection] ❌ Failed to connect using prefix '{env_prefix}': {e}")
        return None

# ==================== 1. Load raw → normalized OLTP tables =========== #

# === DATA READ & EXTRACTION ===

def read_data(source: str, source_type: str = 'auto') -> pd.DataFrame:
    """
    Reads a CSV file from local path, Google Drive ID, or URL.
    
    Parameters:
    - source: file path, GDrive ID, or URL
    - source_type: 'file', 'gdrive', 'url', or 'auto'
    
    Returns:
    - pd.DataFrame
    """
    if source_type == 'auto':
        if source.startswith(('http://', 'https://')):
            source_type = 'url'
        elif os.path.isfile(source):
            source_type = 'file'
        else:
            source_type = 'gdrive'  # fallback
    
    try:
        if source_type == 'file':
            print(f"📂 Loading local file: {source}")
            return pd.read_csv(source)
        
        elif source_type == 'gdrive':
            gdrive_url = f"https://drive.google.com/uc?id={source}"
            print(f"🔗 Loading from Google Drive ID: {source}")
            return pd.read_csv(gdrive_url)

        elif source_type == 'url':
            print(f"🌐 Loading from URL: {source}")
            return pd.read_csv(source)

    except Exception as e:
        raise ValueError(f"❌ Failed to load CSV: {e}")

# df = read_data("data.csv")  # Local file
# df = read_data("1DdmNsrdBRLfzBdgtvzvFHZ7ejFpLlpwW", "gdrive")  # Google Drive
# df = read_data("https://example.com/data.csv")  # URL (auto-detected)
# df.head()

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
    if col in df.columns:
        split_data = df[col].astype(str).str.strip().str.split(pat=' ', n=1, expand=True)

        # Fill missing parts with ''
        split_data = split_data.fillna('')

        # Ensure all new_cols are assigned properly
        for i, new_col in enumerate(new_cols):
            df[new_col] = split_data[i] if i < split_data.shape[1] else ''
        
        print(f"[split_compound_column] ✅ Split '{col}' into {new_cols}")
        #print(df[new_cols].head(3))  # Show preview
    else:
        print(f"[split_compound_column] ⚠️ Column '{col}' not found in DataFrame.")
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
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        name = re.sub(r'[\s\-]+', '_', name)
        return name.lower()

    df.columns = [clean_name(col) for col in df.columns]
    return df

# === 10. Compute Interest based on principal and rate columns ===
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

    # ✅ Use the flexible duplicate remover — all columns by default
    df = remove_duplicates(df)

     # 3. Core transformations — use optional fallbacks
    if "missingvalue_columns" in source_config:
        df = handle_missing_critical(df, source_config["missingvalue_columns"])

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

# ===  GET NORMALIZED OLTP TABLES ===
def split_normalized_tables(
    df: pd.DataFrame,
    table_specs: Dict[str, List[str]],
    critical_columns: Dict[str, List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Splits raw df into normalized tables.
    Drops rows with nulls in critical columns (e.g. PKs).
    
    Parameters:
    - df: preprocessed DataFrame
    - table_specs: table name → list of columns
    - critical_columns: table name → list of required columns (e.g. PKs)
    
    Returns:
    - Dict of table_name → cleaned DataFrame
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
        before = len(table_df)

        # Drop rows with nulls in required (critical) fields
        crits = critical_columns.get(table, [])
        if crits:
            crits = [c for c in crits if c in table_df.columns]
            table_df = table_df.dropna(subset=crits)

        # Drop full duplicates and reset index
        table_df = table_df.drop_duplicates().reset_index(drop=True)
        after = len(table_df)

        print(f"✅ Created '{table}': {after} rows ({before - after} removed: nulls/dupes)")

        tables[table] = table_df

    return tables

# ===== checks if pk exists in splitted tables & creates if it doesnt === #
def generate_index_pks(
    tables: dict,
    pk: dict = None,
    critical_columns: dict = None
) -> dict:
    """
    Infers or generates a single primary key per table.

    Rules:
    - First try: use provided critical_columns (must exist in table and be unique)
    - Special case: 'date_dim' → 'date_id'
    - Then try: singular(table_name) + '_id' (must be unique)
    - Otherwise: generate synthetic key '<singular>_id'

    Parameters:
    - tables: dict of table_name → DataFrame
    - pk: optional existing PK dict
    - critical_columns: dict of table_name → list of critical column names

    Returns:
    - Updated pk dict
    """
    pk = pk or {}
    critical_columns = critical_columns or {}

    for table_name, df in tables.items():
        if table_name in pk and pk[table_name]:
            continue

        # 1. Special case: date_dim → date_id
        if table_name == "date_dim" and "date_id" in df.columns and df["date_id"].is_unique:
            pk[table_name] = ["date_id"]
            print(f"✅ Inferred PK for 'date_dim': date_id")
            continue

        # 2. Use critical_columns if provided
        criticals = critical_columns.get(table_name, [])
        for col in criticals:
            if col in df.columns and df[col].is_unique:
                pk[table_name] = [col]
                print(f"✅ Inferred PK for '{table_name}' from critical_columns: {col}")
                break
        if table_name in pk:
            continue

        # 3. Singular name rule: <table>[:-1]_id
        singular_name = table_name[:-1] if table_name.endswith("s") else table_name
        expected_col = f"{singular_name}_id"

        if expected_col in df.columns and df[expected_col].is_unique:
            pk[table_name] = [expected_col]
            print(f"✅ Inferred PK for '{table_name}': {expected_col}")
            continue

        # 4. Generate synthetic PK
        synthetic_col = f"{singular_name}_id"
        print(f"⚠️ No valid PK found for '{table_name}'. Generating synthetic: '{synthetic_col}'")

        df = df.reset_index(drop=True)
        df[synthetic_col] = range(1, len(df) + 1)
        df = df[[synthetic_col] + [col for col in df.columns if col != synthetic_col]]
        df[synthetic_col] = df[synthetic_col].astype("Int64")

        tables[table_name] = df
        pk[table_name] = [synthetic_col]

    return pk

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

# ===== link generated pk to a table (optional) === #
def apply_configured_foreign_keys(
    tables: dict,
    pk_dict: dict,
    foreign_keys: list,
    return_fk_dict: bool = False
) -> Optional[Dict[str, List[Tuple[str, str, str]]]]:
    """
    Adds FK placeholder columns from config, and optionally returns fk_dict.

    Returns:
    - fk_dict: {to_table: [(fk_column, from_table, from_col), ...]}
    """
    fk_dict = {}

    for fk_spec in foreign_keys:
        from_table = fk_spec["from"]
        to_table = fk_spec["to"]
        fk_name = fk_spec.get("fk_name")

        if from_table not in tables or to_table not in tables:
            print(f"⚠️ FK skipped — missing table(s): {from_table} or {to_table}")
            continue
        if from_table not in pk_dict or not pk_dict[from_table]:
            print(f"⚠️ FK skipped — no PK found for '{from_table}'")
            continue

        pk_col = pk_dict[from_table][0]
        fk_col = fk_name or pk_col

        if fk_col in tables[to_table].columns:
            print(f"ℹ️ FK '{fk_col}' already exists in '{to_table}' — skipping.")
        else:
            tables[to_table][fk_col] = pd.Series(dtype="Int64")
            print(f"✅ Added FK column '{fk_col}' to '{to_table}' (→ {from_table}.{pk_col})")

        # Add to FK dict
        fk_dict.setdefault(to_table, []).append((fk_col, from_table, pk_col))

    return fk_dict if return_fk_dict else None


# ============================================= 2. Build and populate dim_date (DW:IF OLAP CONSIDERED)================================ # 
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

# ============================3. Apply date mapping on OLTP tables ============================= #
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


# ============================4. PK FK ,SK INFERENCING: pk_dict, fk_dict, sk_dict, report = infer_keys_extended(oltp_tables); ============================= #
def infer_foreign_keys_from_pk_dict(
    tables: Dict[str, pd.DataFrame],
    pk_dict: Dict[str, List[str]]
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Infers foreign keys by matching *_id columns against primary keys of other tables.
    
    Returns:
    - fk_dict: table -> list of (fk_col, ref_table, ref_col)
    """
    fk_dict = {}
    pk_lookup = {pk_col: table for table, pk_cols in pk_dict.items() for pk_col in pk_cols}

    for table_name, df in tables.items():
        fk_list = []
        for col in df.columns:
            if col.endswith("_id") and col in pk_lookup:
                ref_table = pk_lookup[col]
                if ref_table != table_name:  # don't refer to self
                    fk_list.append((col, ref_table, col))
        if fk_list:
            fk_dict[table_name] = fk_list

    return fk_dict

def infer_keys_extended(tables: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, str, str]]], Dict[str, List[str]], str]:
    primary_keys, foreign_keys, surrogate_keys = {}, {}, {}
    report_lines = []
    table_name_roots = {
        tbl: tbl.lower().replace("_df", "").replace("dim_", "").replace("fact_", "").rstrip("s")
        for tbl in tables.keys()
    }
    report_lines.append("\n🔍 **Extended Key Inference Report**\n")
    report_lines.append("📌 **Primary Keys**")
    for table_name, df in tables.items():
        root = table_name_roots[table_name]
        pk_candidates = [col for col in df.columns if col.endswith("_id") and root in col.lower() and df[col].is_unique]
        if len(pk_candidates) == 1:
            primary_keys[table_name] = pk_candidates[0]
            report_lines.append(f"   ✔️ {table_name:<20} → {pk_candidates[0]}")
        elif len(pk_candidates) > 1:
            report_lines.append(f"   ⚠️ {table_name:<20} → Multiple PKs: {pk_candidates}")
        else:
            report_lines.append(f"   ❌ {table_name:<20} → No primary key detected")
    pk_lookup = {pk: tbl for tbl, pk in primary_keys.items()}
    report_lines.append("\n📎 **Foreign Keys**")
    for table_name, df in tables.items():
        fk_list = [(col, pk_lookup[col], col) for col in df.columns if col.endswith("_id") and col in pk_lookup and pk_lookup[col] != table_name]
        if fk_list:
            foreign_keys[table_name] = fk_list
            for col, ref_table, ref_col in fk_list:
                report_lines.append(f"   🔗 {table_name:<20} → {col} → {ref_table}.{ref_col}")
        elif table_name not in primary_keys:
            report_lines.append(f"   ℹ️ {table_name:<20} → No PK or FK")
        else:
            report_lines.append(f"   ℹ️ {table_name:<20} → Has PK, no FKs")
    report_lines.append("\n🧬 **Surrogate Key Candidates**")
    for table_name, df in tables.items():
        surrogate_keys[table_name] = [col for col in df.columns if col.endswith('_id') and pd.api.types.is_integer_dtype(df[col])]
        if surrogate_keys[table_name]:
            report_lines.append(f"   🧬 {table_name:<20} → {', '.join(surrogate_keys[table_name])}")
    report_lines.append(f"\n✅ **Inference Summary**")
    report_lines.append(f"   ➤ Total Tables: {len(tables)}")
    report_lines.append(f"   ➤ Primary Keys: {len(primary_keys)}")
    report_lines.append(f"   ➤ Foreign Keys: {sum(len(v) for v in foreign_keys.values())}")
    report_lines.append(f"   ➤ Surrogate Keys: {len(surrogate_keys)} tables evaluated\n")
    return primary_keys, foreign_keys, surrogate_keys, "\n".join(report_lines)

# ===  Table Sorting === #
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

# === CREATE SCHEMA & TABLES (OLTP)==== #

def generate_sql_type(dtype: str) -> str:
    mapping = {
        "object": "TEXT",
        "int64": "INTEGER", "Int64": "INTEGER",
        "float64": "DOUBLE PRECISION",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP"
    }
    return mapping.get(dtype, "TEXT")

def ensure_schema_and_tables(
    conn,
    schema: str,
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, List[str]],
    foreign_keys: Dict[str, List[Tuple[str, str, str]]],
    surrogate_keys: Dict[str, List[str]] = None
) -> None:
    """Create tables in topological order based on FK dependencies"""
    cur = conn.cursor()
    
    # Create schema
    cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s", (schema,))
    if not cur.fetchone():
        cur.execute(f'CREATE SCHEMA "{schema}"')
        print(f"📁 Created schema: {schema}")
    
    # Get creation order based on dependencies
    sorted_tables = topological_sort_tables(tables, foreign_keys)
    print(f"🔀 Table creation order: {sorted_tables}")
    
    # Create tables in dependency order
    for table_name in sorted_tables:
        df = tables[table_name]
        
        # Skip existing tables
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            )
        """, (schema, table_name))
        if cur.fetchone()[0]:
            print(f"⏩ Table {schema}.{table_name} exists - skipping creation")
            continue
            
        print(f"🧱 Creating {schema}.{table_name}...")
        
        # Column definitions
        cols = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sql_type = generate_sql_type(dtype)
            cols.append(f'"{col}" {sql_type}')
        
        # Add PK constraint
        pk_cols = primary_keys.get(table_name, [])
        if pk_cols:
            quoted_pk = [f'"{c}"' for c in pk_cols]
            cols.append(f'PRIMARY KEY ({", ".join(quoted_pk)})')
            print(f"🔑 Adding PK: {pk_cols}")
        
        # Add FK constraints
        for fk_col, ref_table, ref_col in foreign_keys.get(table_name, []):
            cols.append(
                f'FOREIGN KEY ("{fk_col}") '
                f'REFERENCES "{schema}"."{ref_table}" ("{ref_col}")'
            )
            print(f"🔗 Adding FK: {fk_col} → {ref_table}.{ref_col}")
        
        # Execute creation
        create_sql = f'CREATE TABLE "{schema}"."{table_name}" (\n  ' + ',\n  '.join(cols) + '\n)'
        try:
            cur.execute(create_sql)
            print(f"✅ Created {table_name}")
        except Exception as e:
            conn.rollback()
            print(f"❌ Failed to create {table_name}: {str(e)}")
            raise
    
    conn.commit()
    cur.close()

# ==== Load Data into OLTP Tables (staging) ====#Ensure all FKs (customer_id, account_id, *_date_id) are present and non-null where expected
def upsert_dataframe_to_table(
    conn,
    df: pd.DataFrame,
    table_name: str,
    schema: str = "public",
    pk_cols: List[str] = None
) -> None:
    import io
    if df.empty:
        print(f"[UPSERT] ⚠️ Skipping '{schema}.{table_name}' — empty DataFrame")
        return

    try:
        # Step 1: Fetch DB column order and types
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name, is_nullable, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """, (schema, table_name))
        column_info = cur.fetchall()
        cur.close()

        db_columns = [col[0] for col in column_info]
        column_nullability = {col[0]: col[1] for col in column_info}
        column_types = {col[0]: col[2] for col in column_info}

        # Step 2: Clean and align DataFrame
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

        cleaned_df = cleaned_df[db_columns]

        string_cols = [col for col in cleaned_df.columns if cleaned_df[col].dtype == 'object']
        cleaned_df[string_cols] = cleaned_df[string_cols].fillna('')

        # Step 3: Create temp buffer
        buffer = io.StringIO()
        cleaned_df.to_csv(buffer, index=False, header=False, sep="\t", na_rep="\\N")
        buffer.seek(0)

        # Step 4: Insert in try/except
        temp_table = f"temp_{table_name}"
        cur = conn.cursor()
        cur.execute(f'CREATE TEMP TABLE "{temp_table}" (LIKE "{schema}"."{table_name}" INCLUDING ALL);')
        cur.copy_from(buffer, temp_table, sep="\t", null="\\N", columns=db_columns)

        insert_columns = ', '.join(f'"{col}"' for col in db_columns)
        update_assignments = ', '.join(f'"{col}" = EXCLUDED."{col}"' for col in db_columns if col not in pk_cols)

        upsert_sql = f'''
            INSERT INTO "{schema}"."{table_name}" ({insert_columns})
            SELECT {insert_columns} FROM "{temp_table}"
            ON CONFLICT ({', '.join(f'"{col}"' for col in pk_cols)})
            DO UPDATE SET {update_assignments};
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



# ==============================5. OLAP WORKFLOW  FROM OLTP : build empty OLAP tables from the config: === #

def build_empty_olap_tables_from_config(
    dimension_defs: Dict[str, List[str]],
    fact_defs: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """
    Build empty DataFrames representing OLAP table schemas
    based on config-defined dimension and fact table columns.

    Parameters:
    - dimension_defs (dict): e.g. config["olap"]["dimensions"]
    - fact_defs (dict): e.g. config["olap"]["facts"]

    Returns:
    - dict: table_name -> empty DataFrame with defined columns
    """
    olap_tables = {}

    for table_name, columns in {**dimension_defs, **fact_defs}.items():
        olap_tables[table_name] = pd.DataFrame(columns=columns)

    print(f"[build_empty_olap_tables_from_config] ✅ Built {len(olap_tables)} OLAP tables from config.")
    return olap_tables

# ================================7. Build & Load Dimension Tables (dim_*)===================================
def get_olap_tables_from_oltp(
    oltp_tables: Dict[str, Any],
    dimension_defs: Dict[str, list],
    fact_defs: Dict[str, list]
) -> Dict[str, Any]:
    """
    Extract OLAP dimension and fact tables from existing OLTP tables.

    Parameters:
    - oltp_tables: dict of table_name -> DataFrame (OLTP normalized)
    - dimension_defs: dict of dim_table_name -> required column list
    - fact_defs: dict of fact_table_name -> required column list

    Returns:
    - dict of OLAP table_name -> DataFrame
    """
    olap_tables = {}

    # Extract dimensions
    for dim_name, dim_columns in dimension_defs.items():
        matched = False
        for src_table, df in oltp_tables.items():
            if all(col in df.columns for col in dim_columns):
                olap_tables[dim_name] = df[dim_columns].drop_duplicates().reset_index(drop=True)
                print(f"✅ Extracted '{dim_name}' from OLTP table '{src_table}'")
                matched = True
                break
        if not matched:
            print(f"⚠️ Skipped dimension '{dim_name}' — required columns not found in OLTP tables")

    # Extract facts
    for fact_name, fact_columns in fact_defs.items():
        matched = False
        for src_table, df in oltp_tables.items():
            if all(col in df.columns for col in fact_columns):
                olap_tables[fact_name] = df[fact_columns].drop_duplicates().reset_index(drop=True)
                print(f"✅ Extracted '{fact_name}' from OLTP table '{src_table}'")
                matched = True
                break
        if not matched:
            print(f"⚠️ Skipped fact '{fact_name}' — required columns not found in OLTP tables")

    return olap_tables


def build_and_load_all_dimensions(
    conn,
    oltp_tables: Dict[str, pd.DataFrame],
    dimension_defs: Dict[str, List[str]],
    schema: str = "public"
) -> Dict[str, pd.DataFrame]:
    """
    Dynamically builds and loads all dimension tables from config.
    Assigns surrogate keys (SKs) via range().
    
    Parameters:
    - conn: PostgreSQL connection
    - oltp_tables: dict of source OLTP tables (DataFrames)
    - dimension_defs: config["olap"]["dimensions"]
    - schema: target schema (default "zulo_olap")

    Returns:
    - dict of lookup DataFrames: {dim_name: DataFrame with business_id + SK}
    """
    lookup_tables = {}

    for dim_table, columns in dimension_defs.items():
        # Infer business table name (strip 'dim_')
        business_table = dim_table.replace("dim_", "")
        if business_table not in oltp_tables:
            print(f"[DIM LOAD] ⚠️ Skipping '{dim_table}' — source '{business_table}' not found.")
            continue

        source_df = oltp_tables[business_table]

        if not all(col in source_df.columns for col in columns):
            print(f"[DIM LOAD] ❌ Columns missing in source '{business_table}' for '{dim_table}'")
            continue

        # Deduplicate based on business keys
        dim_df = source_df[columns].drop_duplicates().reset_index(drop=True)

        # Create SK
        sk_col = dim_table.replace("dim_", "") + "_sk"
        dim_df[sk_col] = range(1, len(dim_df) + 1)

        # Reorder columns
        dim_df = dim_df[[sk_col] + columns]

        # Load to DB
        upsert_dataframe_to_table(conn, dim_df, table_name=dim_table, schema=schema)
        print(f"[DIM LOAD] ✅ Loaded {len(dim_df)} rows into '{schema}.{dim_table}'")

        # Save lookup
        business_keys = [col for col in columns if col.endswith("_id")]
        lookup_tables[dim_table] = dim_df[business_keys + [sk_col]]

    return lookup_tables

# ================================8. Build & Load Fact Tables (fact_*)===================================

def build_and_load_all_facts(
    conn,
    oltp_tables: Dict[str, pd.DataFrame],
    fact_defs: Dict[str, List[str]],
    dim_lookups: Dict[str, pd.DataFrame],
    schema: str = "public"
) -> None:
    """
    Build and load all fact tables by joining to dimension SK lookups.
    
    Parameters:
    - conn: DB connection
    - oltp_tables: normalized OLTP data
    - fact_defs: config["olap"]["facts"]
    - dim_lookups: mapping from dim_name → lookup DataFrame (business_id → surrogate_sk)
    - schema: target OLAP schema
    """
    for fact_name, fact_columns in fact_defs.items():
        source_name = fact_name.replace("fact_", "")  # e.g., 'fact_loans' → 'loans'
        if source_name not in oltp_tables:
            print(f"[FACT LOAD] ⚠️ Skipping '{fact_name}' — source '{source_name}' missing")
            continue

        fact_df = oltp_tables[source_name].copy()

        # Join with all matching dim lookups
        for dim_name, lkp_df in dim_lookups.items():
            for join_col in lkp_df.columns:
                if join_col in fact_df.columns:
                    sk_col = lkp_df.columns[-1]  # assuming last column is *_sk
                    fact_df = fact_df.merge(lkp_df, on=join_col, how="left")
                    fact_df.drop(columns=[join_col], inplace=True)

        # Keep only columns defined in config
        missing = [col for col in fact_columns if col not in fact_df.columns]
        if missing:
            print(f"[FACT LOAD] ❌ '{fact_name}' missing columns: {missing}")
            continue

        fact_df = fact_df[fact_columns].drop_duplicates().reset_index(drop=True)

        # Add surrogate key for fact row if needed
        fact_sk = fact_name.replace("fact_", "") + "_sk"
        if fact_sk not in fact_df.columns:
            fact_df[fact_sk] = range(1, len(fact_df) + 1)
            fact_df = fact_df[[fact_sk] + fact_columns]  # reorder

        # Load to DB
        upsert_dataframe_to_table(conn, fact_df, table_name=fact_name, schema=schema)
        print(f"[FACT LOAD] ✅ Loaded '{fact_name}' with {len(fact_df)} rows")
        print(f"[FACT JOIN] Joined '{dim_name}' on '{join_col}' → added '{sk_col}'")

# ================================9. Validation & QA===================================
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


# ================================10. (Optional) Create Views, Indexes, & Optimize===================================
#--- indexing

def generate_indexes_on_sk_and_date_ids(
    conn,
    tables: Dict[str, pd.DataFrame],
    schema: str = "public"
) -> None:
    """
    Automatically creates indexes for *_sk and *_date_id columns in a given schema.
    Only uses DataFrame metadata (no DB reflection).
    """
    cur = conn.cursor()
    created = []

    for table_name, df in tables.items():
        for col in df.columns:
            if col.endswith("_sk") or col.endswith("_date_id"):
                index_name = f"idx_{table_name}_{col}"
                stmt = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{schema}"."{table_name}"("{col}");'
                try:
                    cur.execute(stmt)
                    created.append(index_name)
                except Exception as e:
                    print(f"[INDEX] ❌ {index_name}: {e}")
    conn.commit()
    cur.close()

    print(f"\n[INDEX] ✅ Created {len(created)} indexes:")
    for idx in created:
        print(f"   - {idx}")

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


# ================================11. Automate the Pipeline===================================



# ================================12. DATA EXPORT =============================================
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



# etl_visual_helpers.py

def plot_before_after_counts(raw_df: pd.DataFrame, normalized_tables: dict):
    """
    Visualize unique counts in raw_df vs row counts in normalized OLTP tables.
    """
    raw_counts = raw_df.nunique().sort_values(ascending=False)
    normalized_counts = {name: len(df) for name, df in normalized_tables.items()}

    df_raw = pd.DataFrame({"column_or_table": raw_counts.index, "unique_count": raw_counts.values})
    df_norm = pd.DataFrame({"column_or_table": list(normalized_counts.keys()), "row_count": list(normalized_counts.values())})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=df_raw.head(15), x="unique_count", y="column_or_table", ax=axes[0], palette="Blues_d")
    axes[0].set_title("🔍 Top Unique Counts in Raw Data")
    axes[0].set_xlabel("Unique Values")
    axes[0].set_ylabel("Column")

    sns.barplot(data=df_norm, x="row_count", y="column_or_table", ax=axes[1], palette="Greens_d")
    axes[1].set_title("🏗️ Rows per Normalized OLTP Table")
    axes[1].set_xlabel("Row Count")
    axes[1].set_ylabel("Table")

    plt.tight_layout()
    plt.show()


def compare_unique_distribution(raw_df: pd.DataFrame, columns: list):
    """
    Plot a grid of unique value counts for specific raw_df columns.
    """
    num_cols = len(columns)
    fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(14, 5 * ((num_cols + 1) // 2)))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in raw_df.columns:
            counts = raw_df[col].value_counts().head(20)
            sns.barplot(x=counts.values, y=counts.index, ax=axes[i], palette="viridis")
            axes[i].set_title(f"Top 20: {col}")
            axes[i].set_xlabel("Count")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
