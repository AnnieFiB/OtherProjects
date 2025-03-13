# 5. Outlier Capping with selective printing
    if config['num_cols']:
        print("\n🔧 Applying outlier capping:")
        capped_columns = []
        
        for col in config['num_cols']:
            # Skip if no variability
            if df[col].nunique() == 1:
                continue
                
            # Calculate IQR bounds
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Handle near-constant columns
            if iqr < 1e-6:
                continue
                
            # Calculate caps
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Find outliers
            lower_outliers = (df[col] < lower_bound).sum()
            upper_outliers = (df[col] > upper_bound).sum()
            total_outliers = lower_outliers + upper_outliers
            
            # Only cap and print if outliers exist
            if total_outliers > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"   - {col}: Capped {total_outliers} outliers "
                      f"({lower_outliers} low, {upper_outliers} high)")
                capped_columns.append(col)
        
        if not capped_columns:
            print("   - No outliers detected in numerical columns")
   



 # 6. Encode categoricals
    if config['binary_cols']:
        # Check for binary columns with identical values
        valid_binary_cols = [
            col for col in config['binary_cols']
            if df[col].nunique() > 1
        ]
        
        if valid_binary_cols:
            le = LabelEncoder()
            for col in valid_binary_cols:
                df[col] = le.fit_transform(df[col])
            print(f"✅ Binary encoded: {valid_binary_cols}")

    # 7. Scale numericals
    if config['num_cols']:
        scaler = StandardScaler()
        df[config['num_cols']] = scaler.fit_transform(df[config['num_cols']])
        print("✅ Scaled numerical columns")
    
    print("\n🎉 Cleaning complete!")
    print(f"Final shape: {df.shape}")