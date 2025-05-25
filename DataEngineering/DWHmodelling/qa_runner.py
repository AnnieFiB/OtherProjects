import pandas as pd
from typing import Tuple, List, Optional, Dict  

def qa_runner(  
    oltp_tables: Dict[str, pd.DataFrame],
    dim_lookups: Dict[str, pd.DataFrame],
    fact_tables: Dict[str, pd.DataFrame],
    checks: Dict[str, Dict[str, str]]
) -> None:
    """
    Runs QA validation on OLTP vs OLAP pipeline data.
    
    Parameters:
    - oltp_tables: raw or normalized OLTP DataFrames
    - dim_lookups: SK mappings (e.g., customer_id → customer_sk)
    - fact_tables: OLAP fact DataFrames (already joined)
    - checks: dict with validation rules for each fact
    """
    print("\n🧪 STARTING QA CHECKS\n" + "="*30)
    
    for fact_name, rules in checks.items():
        print(f"\n📊 FACT: {fact_name}")
        fact_df = fact_tables.get(fact_name)
        if fact_df is None:
            print("⚠️ Fact table not found.")
            continue

        # 1. Row count vs source
        source_name = fact_name.replace("fact_", "")
        source_df = oltp_tables.get(source_name)
        if source_df is not None:
            print(f"   ➤ Row count: OLTP = {len(source_df)} | OLAP = {len(fact_df)}")

        # 2. Null checks for SKs and date IDs
        for col in rules.get("check_not_null", []):
            nulls = fact_df[col].isna().sum()
            print(f"   🔍 NULL check on '{col}': {nulls} missing")

        # 3. Orphan FK checks
        for sk_col, dim_name in rules.get("fk_checks", {}).items():
            dim_df = dim_lookups.get(dim_name)
            if dim_df is None:
                print(f"   ⚠️ Dimension '{dim_name}' not available.")
                continue
            valid_keys = dim_df[dim_df.columns[-1]]  # assume SK is last
            orphans = ~fact_df[sk_col].isin(valid_keys)
            print(f"   🔗 Orphan check: {orphans.sum()} '{sk_col}' not found in '{dim_name}'")

    print("\n✅ QA CHECKS COMPLETE\n" + "="*30)
