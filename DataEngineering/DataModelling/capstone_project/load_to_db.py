import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import psycopg2.extensions
from psycopg2 import sql
from dotenv import load_dotenv
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)
psycopg2.extensions.register_adapter(np.float64, psycopg2._psycopg.AsIs)

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