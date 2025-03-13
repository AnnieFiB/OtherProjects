def data_cleaning(df, irrelevant_columns=[], binary_cols=[], multi_level_cols=[], 
                 num_cols=[], date_cols=[], column_inconsistencies={}, 
                 missing_value_strategy="drop", fill_value=None):
    """
    Perform comprehensive data cleaning with step-by-step progress messages.
    Returns cleaned DataFrame and prints status updates.
    """
    # Track initial state
    original_shape = df.shape
    
    # 1. Handle missing values
    if missing_value_strategy == "drop":
        initial_null_count = df.isnull().sum().sum()
        df = df.dropna()
        print(f"✅ Missing values handled: Dropped {initial_null_count - df.isnull().sum().sum()} null entries")
    elif missing_value_strategy == "fill":
        fill_count = df.isnull().sum().sum()
        df = df.fillna(fill_value)
        print(f"✅ Missing values handled: Filled {fill_count} null entries")
    
    # 2. Remove irrelevant columns
    cols_before = set(df.columns)
    df = df.drop(columns=irrelevant_columns, errors='ignore')
    removed_cols = list(cols_before - set(df.columns))
    print(f"✅ Columns removed: {removed_cols if removed_cols else 'None'}")

    # 3. Handle date columns
    if date_cols:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"✅ Date columns processed: {date_cols}")
    else:
        print("✅ No date columns to process")

    # 4. Remove duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"✅ Duplicates removed: {dup_count} rows")

    # 5. Fix inconsistencies
    if column_inconsistencies:
        for col, valid_values in column_inconsistencies.items():
            invalid_count = (~df[col].isin(valid_values)).sum()
            df[col] = df[col].apply(lambda x: x if x in valid_values else None)
            print(f"✅ {col} standardized: {invalid_count} invalid values cleaned")
    else:
        print("✅ No inconsistencies to resolve")

    # 6. Encode categorical variables
    if binary_cols:
        label_enc = LabelEncoder()
        for col in binary_cols:
            df[col] = label_enc.fit_transform(df[col])
        print(f"✅ Binary encoded: {binary_cols}")
    else:
        print("✅ No binary columns to encode")
        
    if multi_level_cols:
        df = pd.get_dummies(df, columns=multi_level_cols, drop_first=True)
        print(f"✅ Multi-level encoded: {multi_level_cols}")
    else:
        print("✅ No multi-level columns to encode")

    # 7. Scale numerical features
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"✅ Numerical scaled: {len(num_cols)} columns")
    else:
        print("✅ No numerical columns to scale")

    # 8. Handle outliers
    if num_cols:
        original_rows = df.shape[0]
        for col in num_cols:
            df = df[(np.abs(stats.zscore(df[col])) < 3)]
        removed_outliers = original_rows - df.shape[0]
        print(f"✅ Outliers handled: {removed_outliers} rows removed")
    else:
        print("✅ No numerical columns for outlier handling")

    # Final report
    print("\n🔷 Cleaning Summary:")
    print(f"Original shape: {original_shape}")
    print(f"New shape: {df.shape}")
    print(f"Total change: {original_shape[0] - df.shape[0]} rows, {original_shape[1] - df.shape[1]} cols removed")
    
    return df