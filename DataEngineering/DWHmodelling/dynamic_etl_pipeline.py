# main.py

import sys
#import argparse
from db_utils import *
from sourceconfig import ETL_CONFIG
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML
import io, builtins

# main extraction function
def select_and_load_source(etl_config: dict):
    """
    Interactive dropdown to select and load a data source from ETL_CONFIG.
    Handles local, GDrive, URL, or Kaggle sources.
    Sets raw_df, dataset_key, and cfg globally.
    """
    
    output = widgets.Output()
    result = {'selected_source': None, 'raw_df': None}

    def on_change(change):
        output.clear_output()
        with output:
            selected = change.new
            if selected == 'all':
                print("Please select a valid data source.")
                return

            config = etl_config['data_sources'][selected]
            result['selected_source'] = selected

            try:
                if config.get('type') == 'kaggle':
                    print("🔍 Use the dropdown below to load a file from Kaggle...")
                    fetch_kaggle_dataset_by_path(config['path'])  # Handles setting globals inside
                else:
                    df = read_data(config['path'], config.get('type', 'auto'))

                    result['raw_df'] = df
                    result['selected_source'] = selected

                    builtins.raw_df = df
                    builtins.dataset_key = selected
                    builtins.cfg = etl_config["data_sources"][selected]

                    print(f"✅ Loaded: {selected}")
                    print("📊 Shape:", df.shape)
                    display(df.head(3))

                    buf = io.StringIO()
                    df.info(buf=buf)
                    display(HTML(f"<pre>{buf.getvalue()}</pre>"))

                    print("✅ You can now access raw_df, dataset_key, and cfg globally.")
            except Exception as e:
                print(f"❌ Error: {e}")

    dropdown = widgets.Dropdown(
        options=['all'] + list(etl_config['data_sources'].keys()),
        description='Data Source:',
    )
    dropdown.observe(on_change, names='value')

    display(dropdown, output)
    return result

# Main ETL function for processing,splitting, date mapping a single data source
def transform_oltp(
    dataset_key: str,
    cfg: Dict,
    raw_df: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]], Dict[str, List[Tuple[str, str, str]]], Dict[str, List[str]]]:

    print("\n" + " " * 2 + "*" * 25)
    print(f"\n🔁 Processing {dataset_key}...")

    source_cfg = cfg["oltp"]

    # Step 1: Clean and derive fields
    _, processed_df = process_data_source(raw_df, dataset_key, source_cfg)

    # Step 2: Pre-split raw data (temporary split before final sort)
    print("\n" + " " * 2 + "*" * 25)
    print("\n🔄 Pre-splitting raw data into temporary tables...")
    temp_tables = split_normalized_tables(
    df=processed_df,
    table_specs=source_cfg["table_mapping"],
    critical_columns = source_cfg.get("dedup_columns", source_cfg.get("critical_columns", {}))
    )

    # Step 3: Generate date_dim early if OLAP enabled
    print("\n" + " " * 2 + "*" * 25)
    print("\n📅 Generating and mapping date dimension...")
    date_dim = None
    if "olap" in cfg.get("pipelines", {}) and source_cfg.get("date_mapping"):
        odate_columns = source_cfg.get("odate_columns", [])
        date_input_df = extract_date_columns_from_temp(temp_tables, odate_columns)

        if not date_input_df.empty:
            date_dim = generate_date_dim(date_input_df, odate_columns)
            if not date_dim.empty:
                temp_tables["date_dim"] = date_dim
                temp_tables = apply_configured_date_mapping(
                    temp_tables, date_dim,
                    source_cfg["date_mapping"], date_key="date_id"
                )
                print(f"✅ Mapped {len(date_dim)} date keys.")
            else:
                print("⚠️ Date dimension is empty after generation.")
        else:
            print("⚠️ No valid date columns found for generating date_dim.")
    else:
        print("ℹ️ Skipping date_dim generation. OLAP pipeline not enabled or config missing.")

        
    # Step 4: Generate PKs per temp table
    print("\n" + " " * 2 + "*" * 25)
    print("\n🔑 Generating primary keys for temporary tables...")
    pk_dict = {}
    for tbl, df in temp_tables.items():
        df, pk_dict = generate_index_pks(
            df=df,
            table_mapping={tbl: list(df.columns)},
            pk=pk_dict,
            critical_columns=source_cfg.get("critical_columns", {})
        )
        temp_tables[tbl] = df
        print(f"✅ PKs generated for '{tbl}': {pk_dict.get(tbl, [])}")

    # Step 5: Apply foreign key logic and optionally inject FK values
    print("\n" + " " * 2 + "*" * 25)
    print("\n🔗 Enforcing configured foreign keys and injecting missing values if needed...")
    temp_tables, fk_dict = enforce_foreign_keys(
    temp_tables=temp_tables,
    pk_dict=pk_dict,
    foreign_keys=source_cfg.get("foreign_keys", []),
    return_fk_dict=True
    )

    print(f"✅ Enforced {sum(len(v) for v in fk_dict.values())} foreign key relationships.")

    # Step 6: Topological sort of tables based on FK relationships
    print("\n" + " " * 2 + "*" * 25)
    sorted_order = topological_sort_tables(temp_tables, fk_dict)
    print(f"📋 Table load order resolved: {sorted_order}")
    
    # Step 7: Final OLTP tables (no further dedup or modification needed)
    print("\n" + " " * 2 + "*" * 25)
    oltp_tables = {tbl: temp_tables[tbl] for tbl in sorted_order if tbl in temp_tables}
    print(f"✅ Normalized into {len(oltp_tables)} OLTP tables.")

    # Step 8: Add surrogate keys
    print("\n" + " " * 2 + "*" * 25)
    sk_dict = {}
    if "olap" in cfg.get("pipelines", {}):
        print("\n🧬 Adding surrogate keys...")
        for table_name, df in oltp_tables.items():
            sk_col = f"{table_name}_sk"
            if sk_col not in df.columns:
                df[sk_col] = range(1, len(df) + 1)
                print(f"   ➕ {sk_col} added to {table_name}")
            sk_dict[table_name] = [sk_col]
            oltp_tables[table_name] = df
    else:
        print("ℹ️ Skipping SK generation — OLAP pipeline not enabled.")

    # Step 9: Convert PKs to Int64
    print("\n" + " " * 2 + "*" * 25)
    print("\n🔢 Converting PKs to Int64 where applicable...")
    convert_pk_column_to_int(oltp_tables, pk_dict)

    print(f"\n✅ OLTP transformation complete: {len(oltp_tables)} tables ready.")
    
    return oltp_tables, pk_dict, fk_dict, sk_dict

# Main ETL function for loading and validating OLTP & OLAP tables
def run_dynamic_etl_pipeline(
    conn,
    dataset_key: str,
    raw_df: pd.DataFrame,
    cfg: dict,
    oltp_tables: Optional[Dict[str, pd.DataFrame]] = None,
    pk_dict: Optional[Dict[str, List[str]]] = None,
    fk_dict: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
    sk_dict: Optional[Dict[str, List[str]]] = None
) -> Dict[str, any]:
    """
    Executes the complete ETL pipeline: OLTP → OLAP
    - Enforces PKs, FKs, SKs
    - Applies schema creation and loading
    - Builds dimensions and facts with surrogate key integrity
    """
    logging.info(f"🚀 Starting ETL pipeline for: {dataset_key}")
    status = {"start_time": datetime.now().isoformat(), "stages": {}, "success": True}

    oltp_schema = cfg["oltp"].get("schema", "oltp")
    olap_enabled = "olap" in cfg.get("pipelines", {})
    olap_cfg = cfg.get("olap", {})
    olap_schema = olap_cfg.get("schema", "olap")

    try:
        # Step 1: OLTP transformation
        if oltp_tables is None:
            print("🔁 Transforming dataset into OLTP tables...")
            oltp_tables, pk_dict, fk_dict, sk_dict = transform_oltp(dataset_key, cfg, raw_df)
        else:
            print("\n oltp_tables exist, proceeding with schema creation")

        # Step 2: Create OLTP schema with enforced PKs and FKs
        print("\n" + " " * 2 + "*" * 25)
        print("🧱 Creating OLTP schema and tables...")
        create_oltp_schema_and_tables(
            conn,
            schema=oltp_schema,
            tables=oltp_tables,
            primary_keys=pk_dict,
            foreign_keys=fk_dict
        )
        status["stages"]["oltp_schema"] = "✅ OLTP schema and tables created"

        # Step 3: Load OLTP data (FK order)
        print("\n" + " " * 2 + "*" * 25)
        print("📥 Loading OLTP data into DB...")
        sorted_tables = topological_sort_tables(oltp_tables, fk_dict)
        print(f"🔁 Insert Order: {sorted_tables}")
        for table in sorted_tables[::-1]:
            df = oltp_tables[table]
            upsert_dataframe_to_table(
                conn,
                df,
                table,
                schema=oltp_schema,
                pk_cols=pk_dict.get(table, []),
                skip_on_fk_violation=True
            )
            print(f"   ✅ {table} ({len(df)} rows)")
        status["stages"]["oltp_data_loaded"] = "✅ OLTP data loaded"
        print("\n" + " " * 2 + "*" * 25)

        # Step 4: OLAP pipeline
        if olap_enabled:
            proceed = input("🔨 Proceed with OLAP pipeline? (y/n): ").strip().lower()
            if proceed != "y":
                status["stages"]["olap_skipped"] = "⏭️ Skipped by user"
                status["end_time"] = datetime.now().isoformat()
                return status

            print("\n" + " " * 2 + "*" * 25)
            print("📊 OLAP: Creating dimensions, facts, and keys...")
            dim_defs = olap_cfg.get("dimensions", {})
            fact_defs = olap_cfg.get("facts", {})
            olap_fk_config = cfg.get("olap_foreign_keys", {})

            # Step 4a: Create schema if not exists
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{olap_schema}";')
                conn.commit()
            status["stages"]["olap_schema"] = f"✅ OLAP schema '{olap_schema}' ensured"

            # Step 4b: Copy date_dim from OLTP to OLAP schema
            if "date_dim" in oltp_tables:
                print("🗓️ Copying date_dim to OLAP...")
                copy_date_dim_to_olap(conn, oltp_tables, olap_schema)
            else:
                print("⚠️ 'date_dim' not found in OLTP — skipping copy to OLAP.")

            # Step 4c:Building and loading dimensions tables 
            print("\n" + " " * 2 + "*" * 25)
            print("\n Building and Loading all dim tables from OLTP")
            dim_lookups = build_and_load_all_dimensions(
                conn, oltp_tables, dim_defs, schema=olap_schema
            )
            status["stages"]["olap_dimensions"] = "✅ Dimensions created"

            # Step 4b: Building and loading fact tables
            print("\n" + " " * 2 + "*" * 25)
            print("\n📦 Building and Loading all fact tables from OLTP & OLAP...")
            if fact_defs:
                fact_lookups = build_and_load_all_facts(
                    conn=conn,
                    oltp_tables=oltp_tables,
                    fact_config=fact_defs,
                    dim_lookups=dim_lookups,
                    schema=olap_schema
                )
                status["stages"]["olap_facts"] = "✅ Facts created"
            else:
                print("⚠️ No OLAP fact configuration found — skipping fact generation.")
                fact_lookups = {}


            # Step 4c: Enforce OLAP foreign keys
            print("\n" + " " * 2 + "*" * 25)
            if "olap_foreign_keys" in cfg:
                print("\n🔗 Enforcing OLAP foreign keys...")
                enforce_olap_foreign_keys(
                    conn=conn,
                    schema=olap_schema,
                    olap_fk_config=cfg["olap_foreign_keys"]
                )

            # Step 4d: Indexing
            print("\n" + " " * 2 + "*" * 25)
            print("⚙️ Indexing SKs and dates in OLAP...")
            generate_indexes_on_sk_and_date_ids(
                conn,
                {**dim_lookups, **fact_lookups},
                schema=olap_schema
            )
            status["stages"]["olap_indexes"] = "✅ OLAP indexes created"
            print("\n" + " " * 2 + "*" * 25)

            logging.info("✅ ETL pipeline completed successfully.")
            print("✅ All ETL stages complete.")

    except Exception as e:
        logging.error(f"❌ ETL pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        status["success"] = False

    print("\n" + " " * 2 + "*" * 25)
    status["end_time"] = datetime.now().isoformat()
    return status



