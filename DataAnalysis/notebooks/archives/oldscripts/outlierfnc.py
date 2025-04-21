import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Detect outliers in a single column
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Cap outliers in a single column
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    capped = data.copy()
    capped[column] = capped[column].clip(lower=lower_bound, upper=upper_bound)
    return capped

# Detect outliers in all numeric columns
def detect_outliers_all(df):
    outlier_summary = {}
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if df[col].nunique() <= 2:
            continue  # skip binary or constant columns
        outliers = detect_outliers(df, col)
        outlier_summary[col] = len(outliers)
    
    return pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['outlier_count'])


# Cap outliers in all numeric columns
def cap_outliers_all(df):
    df_capped = df.copy()
    numeric_cols = df_capped.select_dtypes(include='number').columns

    for col in numeric_cols:
        df_capped = cap_outliers(df_capped, col)
    
    return df_capped

# Optional: plot before vs after for any numeric column
def plot_outlier_distributions(original, capped, column):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.boxplot(y=original[column], ax=axs[0])
    axs[0].set_title(f'Original: {column}')
    
    sns.boxplot(y=capped[column], ax=axs[1])
    axs[1].set_title(f'Capped: {column}')
    
    plt.tight_layout()
    plt.show()
