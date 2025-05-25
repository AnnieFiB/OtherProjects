import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

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

    print(f"🔍 Preparing to upsert {len(df)} rows into {schema}.{table_name}...")

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

    print(f"📝 SQL Statement:\n{insert_sql}")
    print(f"📌 Conflict Columns: {conflict_columns}")
    print(f"📌 Update Columns: {update_columns}")

    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, values)
        conn.commit()
        print(f"✅ {len(df)} records upserted into {schema}.{table_name}")
    except Exception as e:
        print(f"❌ Failed to upsert into {schema}.{table_name}: {e}")
        conn.rollback()


def load_all_known_tables(conn,
                          customers_df, products_df, payments_df, locations_df, orders_df,
                          dim_customer, dim_product, dim_payment, dim_location, dim_date, fact_sales,
                          fact_lookup):
    try:
        upsert_from_df(conn, customers_df, 'customers', ['customer_id'], schema='oltp')
        upsert_from_df(conn, products_df, 'products', ['product_id'], schema='oltp')
        upsert_from_df(conn, payments_df, 'payments', ['payment_id'], schema='oltp')
        upsert_from_df(conn, locations_df, 'locations', ['location_id'], schema='oltp')
        upsert_from_df(conn, orders_df, 'orders', ['order_id'], schema='oltp')

        upsert_from_df(conn, fact_lookup, 'fact_lookup', ['order_sk'], schema='olap')
        upsert_from_df(conn, dim_customer, 'dim_customer', ['customer_sk'], schema='olap')
        upsert_from_df(conn, dim_product, 'dim_product', ['product_sk'], schema='olap')
        upsert_from_df(conn, dim_payment, 'dim_payment', ['payment_sk'], schema='olap')
        upsert_from_df(conn, dim_location, 'dim_location', ['location_sk'], schema='olap')
        upsert_from_df(conn, dim_date, 'dim_date', ['date_id'], schema='olap')
        upsert_from_df(conn, fact_sales, 'fact_sales', ['order_sk'], schema='olap')

        print("🎉 All tables successfully uploaded.")

    except Exception as e:
        print("❌ Error during table uploads:", e)
        conn.rollback()
