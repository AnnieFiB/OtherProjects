# main.py

import sys
import argparse
from db_utils import *
from config import ETL_CONFIG
import logging
from datetime import datetime


# main extraction function
def select_and_load_source(etl_config: dict):
    """
    Interactive dropdown to select and load a data source from ETL_CONFIG.
    Returns selected source and loaded DataFrame via callback.
    """
    output = widgets.Output()

    result = {
        'selected_source': None,
        'raw_df': None
    }

    def on_change(change):
        output.clear_output()
        with output:
            selected = change.new
            if selected == 'all':
                print("Please select a valid data source.")
                return

            try:
                config = etl_config['data_sources'][selected]
                df = read_data(config['path'], config.get('type', 'auto'))

                result['selected_source'] = selected
                result['raw_df'] = df

                print(f"✅ Loaded: {selected}")
                print("📊 Shape:", df.shape)
                display(df.head(3))

                buf = io.StringIO()
                df.info(buf=buf)
                display(HTML(f"<pre>{buf.getvalue()}</pre>"))

                print("✅ You can now access result['raw_df'], result['selected_source']")
            except Exception as e:
                print(f"❌ Error: {e}")

    dropdown = widgets.Dropdown(
        options=['all'] + list(etl_config['data_sources'].keys()),
        description='Data Source:',
    )
    dropdown.observe(on_change, names='value')

    display(dropdown, output)
    return result  # <- return the container so notebook can track updates


# Main ETL function for processing,splitting, date mapping a single data source
def transform_oltp(
    dataset_key: str,
    cfg: Dict,
    raw_df: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]], Dict[str, List[Tuple[str, str, str]]], Dict[str, List[str]]]:
    """
    Transforms raw data into OLTP-normalized tables:
    - Cleans and derives fields
    - Infers PKs using config-critical columns or generates synthetic
    - Converts PKs to Int64
    - Drops rows where PKs are null
    - Optionally generates date_dim and maps *_date_id
    - Splits into normalized tables (drops nulls/dupes)
    - Applies or infers FK relationships
    - Identifies surrogate key candidates (OLAP)
    """
    print(f"\n🔁 Processing {dataset_key}...")
    source_cfg = cfg["oltp"]

    # Step 1: Preprocessing
    _, processed_df = process_data_source(raw_df, dataset_key, source_cfg)

    # Step 2: Optional Date dimension creation (before normalization)
    date_dim = None
    if "olap" in cfg["pipelines"] and source_cfg.get("date_mapping"):
        print("\n📅 OLAP pipeline active — generating date dimension and mapping *_date_id...")
        date_columns = source_cfg.get("date_columns", [])
        if date_columns:
            date_dim = generate_date_dim(processed_df, date_columns)
            if not date_dim.empty:
                print(f"✅ Date dimension created with {len(date_dim)} unique dates.")
            else:
                print("⚠️ Date dimension is empty — skipping mapping.")
        else:
            print("⚠️ No date_columns defined — skipping date_dim.")
    else:
        print("ℹ️ Skipping date_dim — OLAP pipeline or date mapping not configured.")

    # Step 3: Infer PKs using critical_columns
    print("\n🔑 Inferring or generating primary keys (before split)...")
    pk_dict = {}
    for table_name, columns in source_cfg["table_mapping"].items():
        crits = source_cfg.get("critical_columns", {}).get(table_name, [])
        for col in crits:
            if col in processed_df.columns and processed_df[col].is_unique:
                pk_dict[table_name] = [col]
                print(f"✅ Inferred PK for '{table_name}' from critical column: {col}")
                break
        else:
            print(f"⚠️ No critical PK found for '{table_name}' — will generate after split.")

    # Step 4: Drop rows with nulls in PK columns
    for table, pk_cols in pk_dict.items():
        for pk in pk_cols:
            if pk in processed_df.columns:
                before = len(processed_df)
                processed_df = processed_df.dropna(subset=[pk])
                after = len(processed_df)
                if before != after:
                    print(f"🧹 Dropped {before - after} rows with null '{pk}' (used as PK) in flat dataset.")

   # Step 5: Normalize into OLTP tables (with null/dupe control)
    print("\n🔄 Normalising (3NF) raw data and splitting into OLTP tables...")
    critical_map = {
        tbl: [col for col in source_cfg.get("critical_columns", {}).get(tbl, []) if col in cols]
        for tbl, cols in source_cfg["table_mapping"].items()
    }
    oltp_tables = split_normalized_tables(processed_df, source_cfg["table_mapping"], critical_columns=critical_map)

    # Step 6: Map *_date_id if date_dim exists
    if date_dim is not None and not date_dim.empty:
        oltp_tables["date_dim"] = date_dim
        oltp_tables = apply_configured_date_mapping(
            oltp_tables,
            date_dim,
            source_cfg["date_mapping"],
            date_key="date_id"
        )
        print("✅ Date IDs mapped into OLTP tables.")

    # Step 7: Regenerate missing PKs after split
    print("\n🔁 Revalidating PKs post-split...")
    pk_dict = generate_index_pks(oltp_tables, pk=pk_dict, critical_columns=source_cfg.get("critical_columns", {}))

    # Step 8: Apply or infer FKs
    if "foreign_keys" in source_cfg:
        print("\n🔗 Applying configured foreign key relationships...")
        fk_dict = apply_configured_foreign_keys(
            tables=oltp_tables,
            pk_dict=pk_dict,
            foreign_keys=source_cfg["foreign_keys"],
            return_fk_dict=True
        )
    else:
        print("\n🔍 No FK config provided — inferring FKs from PK structure...")
        fk_dict = infer_foreign_keys_from_pk_dict(oltp_tables, pk_dict)

    # Step 9: Detect Surrogate Keys (OLAP only)
    sk_dict = {}
    if "olap" in cfg["pipelines"]:
        print("\n🧬 Inferring surrogate key candidates (for OLAP schema)...")
        for table_name, df in oltp_tables.items():
            sk_candidates = [
                col for col in df.columns
                if col.endswith("_id") and pd.api.types.is_integer_dtype(df[col])
            ]
            if sk_candidates:
                sk_dict[table_name] = sk_candidates
    else:
        print("ℹ️ Skipping SK detection — OLAP pipeline not enabled.")

    print(f"\n✅ Transformation complete. {len(oltp_tables)} tables ready with PKs and FKs.")
    
    # Step 10: Convert PK columns to Int64 (after split)
    print("\n🔢 Ensuring PK columns are Int64 where applicable...")
    convert_pk_column_to_int(oltp_tables, pk_dict)

    return oltp_tables, pk_dict, fk_dict, sk_dict


# Main ETL function for loading and validating OLTP & OLAP tables
def run_dynamic_etl_pipeline(
    conn,
    dataset_key: str,
    raw_df: pd.DataFrame,
    cfg: dict,
    oltp_tables: Dict[str, pd.DataFrame] ,
    pk_dict: Dict[str, List[str]],  
    fk_dict: Dict[str, List[Tuple[str, str, str]]],  
    sk_dict: Dict[str, List[str]] = None
) -> Dict[str, any]:
    from datetime import datetime
    import logging

    logging.info(f"🚀 Starting ETL pipeline for: {dataset_key}")
    status = {"start_time": datetime.now().isoformat(), "stages": {}, "success": True}

    oltp_schema = cfg["oltp"].get("schema", "oltp")

    try:
        # Step 1: Transform if not provided
        if oltp_tables is None:
            print("🔁 Transforming data...")
            oltp_tables, pk_dict, fk_dict, sk_dict = transform_oltp(dataset_key, cfg, raw_df)
        else:
            # Enforce required parameters when providing pre-transformed data
            if not pk_dict or not fk_dict:
                raise ValueError("Must provide pk_dict and fk_dict when passing oltp_tables")

        # Step 2: Create schema/tables with enforced constraints
        ensure_schema_and_tables(
            conn,
            schema=oltp_schema,
            tables=oltp_tables,
            primary_keys=pk_dict,
            foreign_keys=fk_dict
        )

        # Step 3: Load data in topological order
        for tbl in topological_sort_tables(oltp_tables, fk_dict):
            df = oltp_tables[tbl]
            
            # Validate PK columns exist
            for pk in pk_dict.get(tbl, []):
                if pk not in df.columns:
                    raise ValueError(f"Missing PK column '{pk}' in table '{tbl}'")

            print(f"📌 Loading: {tbl} — PK: {pk_dict.get(tbl)}, FK: {fk_dict.get(tbl, [])}")
            upsert_dataframe_to_table(
                conn,
                df,
                table_name=tbl,
                schema=oltp_schema,
                pk_cols=pk_dict.get(tbl, [])
            )

        print(f"✅ ETL pipeline complete for: {dataset_key}")
        logging.info(f"✅ ETL complete for '{dataset_key}'")

    except Exception as e:
        logging.error(f"❌ ETL failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        status["success"] = False

    status["end_time"] = datetime.now().isoformat()
    return status