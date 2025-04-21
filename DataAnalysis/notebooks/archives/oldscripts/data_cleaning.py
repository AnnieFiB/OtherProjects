import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

def detect_columns(df):
    """Detect column types with clear categories"""
    config = {
        'num_cols': [],
        'binary_cols': [],
        'multi_cat_cols': [],
        'date_cols': [],
        'irrelevant_cols': []
    }

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            config['date_cols'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            config['num_cols'].append(col)
        else:
            unique_vals = df[col].nunique()
            if unique_vals == 1:
                config['irrelevant_cols'].append(col)
            elif unique_vals == 2:
                config['binary_cols'].append(col)
            elif 3 <= unique_vals <= 15:
                config['multi_cat_cols'].append(col)
            else:
                config['irrelevant_cols'].append(col)

    print("🔍 Detected columns:")
    print(f"- Dates: {config['date_cols']}")
    print(f"- Numerical: {config['num_cols']}")
    print(f"- Binary: {config['binary_cols']}")
    print(f"- Multi-category: {config['multi_cat_cols']}")
    print(f"- Irrelevant: {config['irrelevant_cols']}")
    
    return config

def clean_data(df, config, missing_strategy='fill'):
    """
    Clean data with flexible missing value handling
    Parameters:
        missing_strategy: 'fill' (default) or 'drop'
    """
    print(f"\n🚀 Starting cleaning (missing strategy: {missing_strategy})...")
    
    # 1. Remove irrelevant columns
    df = df.drop(columns=config['irrelevant_cols'])
    print(f"✅ Removed {len(config['irrelevant_cols'])} irrelevant columns")
    
    # 2. Process dates
    if config['date_cols']:
        for col in config['date_cols']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"📅 Processed {len(config['date_cols'])} date columns")
    
    # 3. Handle missing values
    original_rows = len(df)
    missing_before = df.isna().sum().sum()
    
    if missing_strategy == 'fill':
        df = df.fillna({
            col: df[col].median() if col in config['num_cols'] else df[col].mode()[0]
            for col in df.columns
        })
        print(f"✅ Filled {missing_before} missing values")
    elif missing_strategy == 'drop':
        df = df.dropna().reset_index(drop=True)
        removed = original_rows - len(df)
        print(f"✅ Dropped {removed} rows with missing values")
    else:
        raise ValueError("Invalid missing_strategy. Use 'fill' or 'drop'")
    
    # 4. Remove duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"✅ Removed {dup_count} duplicate rows")
    


    
    return df
