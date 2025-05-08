import pandas as pd
import numpy as np
import re, os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union,Optional, Tuple
from sqlalchemy import create_engine

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
