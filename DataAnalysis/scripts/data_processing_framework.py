import os
import sys
import kaggle
import pandas as pd
import requests
from zipfile import ZipFile
from io import BytesIO
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway



def fetch_kaggle_dataset(search_query="human resources"):
    """
    Authenticate Kaggle API, search for datasets, download, list available files, and allow user input to select a dataset.

    Parameters:
    search_query (str): The keyword for searching datasets on Kaggle.

    Returns:
    pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    # Dynamically resolve the Kaggle config directory
    venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.myenv'))
    kaggle_config_dir = os.path.join(venv_path, '.kaggle')

    # Set the KAGGLE_CONFIG_DIR environment variable
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir

    # Authenticate with Kaggle API
    kaggle.api.authenticate()

    # Search for datasets related to the query
    search_result = kaggle.api.dataset_list(search=search_query)

    if not search_result:
        print("❌ No datasets found for the search query.")
        return None

    # Limit results to the top 5 datasets
    top_datasets = search_result[:5]

    # Print dataset details and list available files
    print("\n🔹 Available Datasets:")
    dataset_refs = []
    file_info_dict = {}
    
    for i, dataset in enumerate(top_datasets):
        print(f"\nDataset {i + 1}: {dataset.ref} - {dataset.title}")
        dataset_refs.append(dataset.ref)

        # Download the dataset ZIP file into memory
        api_url = f'https://www.kaggle.com/api/v1/datasets/download/{dataset.ref}'
        response = requests.get(api_url, stream=True)
        zip_file = ZipFile(BytesIO(response.content))

        # List all files in the ZIP archive
        print("Files:")
        file_list = []
        for file_info in zip_file.infolist():
            file_name = file_info.filename
            file_size = file_info.file_size
            print(f"  - {file_name} (Size: {file_size} bytes)")
            file_list.append((file_name, file_size, file_info))
        
        # Store file info for the dataset
        file_info_dict[i + 1] = (dataset.ref, file_list)

    # Allow user to select dataset by number
    while True:
        try:
            dataset_index = int(input("\nEnter the number of the dataset you want to use: "))
            
            # Validate dataset index
            if dataset_index < 1 or dataset_index > len(dataset_refs):
                print(f"❌ Invalid dataset number. Please enter a number between 1 and {len(dataset_refs)}.")
                continue
            
            break  # Exit loop if input is valid
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

    # Get the selected dataset and its files
    selected_dataset_ref, file_list = file_info_dict[dataset_index]

    # If there are multiple files, let the user choose one
    if len(file_list) > 1:
        print("\n🔹 Available Files:")
        for j, (file_name, file_size, _) in enumerate(file_list):
            print(f"{j + 1}. {file_name} (Size: {file_size} bytes)")
        
        while True:
            try:
                file_index = int(input("\nEnter the number of the file you want to use: "))
                
                # Validate file index
                if file_index < 1 or file_index > len(file_list):
                    print(f"❌ Invalid file number. Please enter a number between 1 and {len(file_list)}.")
                    continue
                
                break  # Exit loop if input is valid
            except ValueError:
                print("❌ Invalid input. Please enter a valid number.")
        
        selected_file_name, _, selected_file_info = file_list[file_index - 1]
    else:
        # If there's only one file, select it automatically
        selected_file_name, _, selected_file_info = file_list[0]

    # Download the selected dataset ZIP file into memory
    api_url = f'https://www.kaggle.com/api/v1/datasets/download/{selected_dataset_ref}'
    response = requests.get(api_url, stream=True)
    zip_file = ZipFile(BytesIO(response.content))

    # Open the selected file
    file_ext = os.path.splitext(selected_file_name)[1].lower()
    file = zip_file.open(selected_file_name)

    # Load the file based on its type
    if file_ext == ".csv":
        # List of encodings to try (order matters)
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                # Reset file pointer for each attempt
                file.seek(0)
                data = pd.read_csv(file, encoding=encoding)
                print(f"✅ Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"⚠️  Failed to read with {encoding} encoding, trying next...")
                continue
        else:
            raise ValueError(
                "❌ Failed to decode CSV file. Tried encodings: " + 
                ", ".join(encodings_to_try) + 
                "\nPlease manually inspect the file encoding."
            )
    elif file_ext == ".xlsx":
        data = pd.read_excel(file)
    elif file_ext == ".json":
        data = pd.read_json(file)
    else:
        raise ValueError(f"❌ Unsupported file type: {file_ext}")

    print("\n✅ Dataset loaded successfully!")
    print(data.info())

    return data

def detect_columns(df):
    """Detect column types and return results in a DataFrame."""
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
            if unique_vals == 2:
                config['binary_cols'].append(col)
            elif 3 <= unique_vals <= 15:
                config['multi_cat_cols'].append(col)
            else:  # Handles both unique_vals == 1 and >15 cases
                config['irrelevant_cols'].append(col)

    # Convert to DataFrame
    result_df = pd.DataFrame({
        "Category": ["Dates", "Numerical", "Binary", "Multi-Category", "Irrelevant"],
        "Columns": [
            config['date_cols'],
            config['num_cols'],
            config['binary_cols'],
            config['multi_cat_cols'],
            config['irrelevant_cols']
        ],
        "Count": [
            len(config['date_cols']),
            len(config['num_cols']),
            len(config['binary_cols']),
            len(config['multi_cat_cols']),
            len(config['irrelevant_cols'])
        ]
    })
    
    return result_df


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


def generate_autoviz_html_report(folder_name="reports_html", output_filename="AutoViz_Report.html"):
    folder_path = os.path.abspath(folder_name)
    html_path = os.path.join(folder_path, output_filename)

    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist.")
        return None

    svg_files = [f for f in os.listdir(folder_path) if f.endswith(".svg")]
    if not svg_files:
        print("⚠️ No .svg files found in the folder.")
        return None

    html_content = "<html><head><title>AutoViz EDA Report</title></head><body>\n"
    html_content += f"<h1>📊 AutoViz EDA Report</h1>\n"
    html_content += f"<p><b>Generated from folder:</b> {folder_name}</p><hr>\n"

    for filename in sorted(svg_files):
        svg_path = os.path.join(folder_path, filename)
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        html_content += f"<div>{svg_content}</div>\n<hr>\n"

    html_content += "</body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ AutoViz HTML report saved to: {html_path}")
    return html_path


# chi_square_analysis.py

def anova_test_numerical_features(df, numeric_columns, target_column='churn', plot=True):
    '''
    Perform ANOVA test for numerical columns against a binary target variable and plot histograms.
    Converts target to binary 0/1 if it's not numeric.

    Parameters:
    - df: pandas DataFrame
    - numeric_columns: list of numerical column names to test
    - target_column: binary target variable (e.g., 'churn')
    - plot: whether to show distribution histograms

    Returns:
    - results_df: DataFrame with p-values and significance flags
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import f_oneway

    results = []

    # Convert target to binary if not numeric
    if df[target_column].dtype != 'int64' and df[target_column].dtype != 'float64':
        if set(df[target_column].unique()) == {'yes', 'no'}:
            df[target_column] = df[target_column].map({'no': 0, 'yes': 1})
        else:
            raise ValueError(f"Target column '{target_column}' must be binary and convertible to 0/1.")

    for col in numeric_columns:
        if col not in df.columns or df[col].nunique() <= 1:
            continue
        try:
            group1 = df[df[target_column] == 0][col].dropna()
            group2 = df[df[target_column] == 1][col].dropna()
            f_stat, p_value = f_oneway(group1, group2)
            significant = p_value < 0.05
            results.append({
                'feature': col,
                'p_value': round(p_value, 4),
                'significant': significant
            })

            if plot:
                plt.figure(figsize=(8, 4))
                sns.histplot(data=df, x=col, hue=target_column, kde=True, element='step')
                plt.title(f'{col} Distribution by {target_column} (p={p_value:.4f})')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            results.append({
                'feature': col,
                'p_value': None,
                'significant': False,
                'error': str(e)
            })

    return pd.DataFrame(results)



def chi_square_test(df, cat_column, target_column='churn', plot=True):
    '''
    Perform Chi-Square Test of Independence between a categorical feature and the target (e.g., churn).
    
    Parameters:
    - df (pd.DataFrame): The dataset
    - cat_column (str): The name of the categorical feature to test
    - target_column (str): The name of the binary target column (default is 'churn')
    - plot (bool): If True, show a countplot of the feature grouped by target

    Returns:
    - p_value (float): The p-value from the Chi-Square test
    - conclusion (str): Whether the result is statistically significant
    '''
    # Create contingency table
    contingency_table = pd.crosstab(df[cat_column], df[target_column])

    # Perform Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Visualize
    if plot:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=cat_column, hue=target_column)
        plt.title(f'Distribution of {cat_column} by {target_column}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Interpret result
    significance = 0.05
    if p < significance:
        conclusion = f"✅ Significant relationship (p-value = {p:.4f})"
    else:
        conclusion = f"❌ No significant relationship (p-value = {p:.4f})"
    
    return p, conclusion

def chi_square_test_batch(df, cat_columns, target_column='churn', plot=False):
    """
    Run chi-square test on multiple categorical columns against a binary target.

    Parameters:
    - df: DataFrame
    - cat_columns: list of categorical column names
    - target_column: name of the binary target
    - plot: whether to plot each distribution

    Returns:
    - DataFrame summarizing p-values and test significance
    """
    results = []

    for col in cat_columns:
        try:
            p, conclusion = chi_square_test(df, col, target_column, plot=plot)
            results.append({
                'feature': col,
                'p_value': round(p, 4),
                'significant': p < 0.05
            })
        except Exception as e:
            results.append({
                'feature': col,
                'p_value': None,
                'significant': False,
                'error': str(e)
            })

    return pd.DataFrame(results)


def plot_significant_categorical_proportions(df, significant_features, target_column='churn'):
    '''
    Plots proportion bar charts of significant categorical features vs the target (e.g., churn),
    with data labels on each bar segment.

    Parameters:
    - df: pandas DataFrame
    - significant_features: list of features to plot
    - target_column: the target variable to compare against
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt

    for col in significant_features:
        if col not in df.columns:
            continue

        prop_df = pd.crosstab(df[col], df[target_column], normalize='index')
        ax = prop_df.plot(kind='bar', stacked=True, colormap='Set2', figsize=(8, 4))

        # Add data labels
        for p in ax.patches:
            height = p.get_height()
            if height > 0.01:  # Only show labels for visible segments
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_y() + height / 2,
                    f'{height:.1%}',
                    ha='center', va='center',
                    fontsize=9
                )

        plt.title(f'Proportion of {target_column} by {col}')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.legend(title=target_column)
        plt.tight_layout()
        plt.show()


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


def plot_outliers_boxplot(df, title="Outlier Boxplot", figsize=(15, 6)):
    """
    Plots horizontal boxplots for all numeric columns in a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - title (str): Plot title.
    - figsize (tuple): Figure size.
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.empty:
        print("No numeric columns found to plot.")
        return

    # Plot
    plt.figure(figsize=figsize)
    sns.boxplot(data=numeric_df, orient="h")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def winsorize_dataframe(df, numeric_cols=None, limits=[0.01, 0.99]):
    """
    Applies Winsorization to numeric columns in a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame to modify.
    - numeric_cols (list): List of numeric column names to Winsorize. If None, it auto-detects numeric columns.
    - limits (list): Lower and upper quantile limits (e.g., [0.01, 0.99]).

    Returns:
    - pd.DataFrame: DataFrame with Winsorized numeric columns.
    """
    # Detect numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Apply Winsorization to each numeric column
    for col in numeric_cols:
        lower = df[col].quantile(limits[0])
        upper = df[col].quantile(limits[1])
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df
