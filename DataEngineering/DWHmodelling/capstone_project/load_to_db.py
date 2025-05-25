import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import psycopg2.extensions
from psycopg2 import sql
import os, sys
from dotenv import load_dotenv
load_dotenv()
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)
psycopg2.extensions.register_adapter(np.float64, psycopg2._psycopg.AsIs)

def check_and_create_db(target_dbname: str, url_env_var: str):
    """
    Connect to a PostgreSQL admin DB using a URL from an env var,
    and create `target_dbname` if it doesn't exist.

    Parameters:
    - target_dbname (str): Name of the DB to check/create
    - url_env_var (str): Name of the env var that contains the full DB URL (e.g., 'SALES_DB_ADMIN_URL')

    Required:
    - An environment variable named `url_env_var` must exist and hold a valid PostgreSQL URL to an admin DB (e.g., .../postgres)
    """
    load_dotenv()

    admin_db_url = os.getenv(url_env_var)
    if not admin_db_url:
        raise ValueError(f"❌ Environment variable '{url_env_var}' is not set.")

    if not admin_db_url.lower().endswith("/postgres"):
        print("⚠️ Note: Admin DB URL usually ends with '/postgres' for DB management.")

    try:
        conn = psycopg2.connect(admin_db_url)
        conn.autocommit = True
        cur = conn.cursor()

        # Check if database already exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", [target_dbname])
        exists = cur.fetchone()

        if not exists:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_dbname)))
            print(f"✅ Database '{target_dbname}' created.")
        else:
            print(f"ℹ️ Database '{target_dbname}' already exists.")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error while checking/creating DB '{target_dbname}': {e}")


def upsert_from_df(conn, df, table_name, conflict_columns, update_columns=None, schema='public'):
    """
    Upserts a DataFrame into a PostgreSQL table with debug status output.

    Parameters:
    - conn: psycopg2 connection object
    - df: pandas DataFrame to upsert
    - table_name: name of the target table
    - conflict_columns: list of columns to use for ON CONFLICT clause
    - update_columns: list of columns to update on conflict; if None, all except conflict_columns
    - schema: database schema (default is 'public')
    """
    if df is None or df.empty:
        print(f"⚠️ Skipping {schema}.{table_name}: DataFrame is empty or None.")
        return

    print(f"Preparing to upsert {len(df)} rows into {schema}.{table_name}...")

    if update_columns is None:
        update_columns = [col for col in df.columns if col not in conflict_columns]

    columns = list(df.columns)
    values = [tuple(x) for x in df.to_numpy()]
    placeholders = ', '.join(columns)
    conflict_cols = ', '.join(conflict_columns)
    update_stmt = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])

    insert_sql = (
        f"INSERT INTO {schema}.{table_name} ({placeholders})\n"
        f"VALUES %s\n"
        f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_stmt};"
    )

    #print(f" SQL Statement:\n{insert_sql}")
    #print(f" Conflict Columns: {conflict_columns}")
    #print(f" Update Columns: {update_columns}")

    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, values)
        conn.commit()
        print(f"✅ {len(df)} records upserted into {schema}.{table_name}")
    except Exception as e:
        print(f"❌ Failed to upsert into {schema}.{table_name}: {e}")
        conn.rollback()