# Description: This file contains the helper functions for the HR dataset analysis.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import lifelines
from lifelines import CoxPHFitter
# ===============================
# Function to check for anomalies
def data_quality_check(df):
    """
    Perform a data quality check on the given DataFrame, visualize outliers, and return results in a structured format.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.

    Returns:
    dict: A dictionary containing the results of the data quality checks.
    """
    results = {}

    # Check for missing values
    missing_values = df.isnull().sum()
    results['missing_values'] = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values
    })

    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    results['duplicate_rows'] = f"Number of duplicate rows: {duplicate_rows}"

    # Check for unique values in categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    unique_values = {}
    for col in categorical_columns:
        unique_values[col] = df[col].nunique()
    results['unique_values'] = pd.DataFrame({
        'Column': unique_values.keys(),
        'Unique Values': unique_values.values()
    })

    # Check for potential outliers in numerical columns (visualization only)
    numerical_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    if len(numerical_columns) > 0:
        for col in numerical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

            # Plot scatter plot for outliers
            plt.figure(figsize=(8, 6))
            plt.scatter(df.index, df[col], color='blue', label='Normal Data')
            plt.scatter(df[outlier_mask].index, df[col][outlier_mask], color='red', label='Outliers')
            plt.axhline(y=lower_bound, color='green', linestyle='--', label='Lower Bound')
            plt.axhline(y=upper_bound, color='orange', linestyle='--', label='Upper Bound')
            plt.title(f'Outliers in {col}')
            plt.xlabel('Index')
            plt.ylabel(col)
            plt.legend()
            plt.show()
    else:
        print("No numerical columns found for outlier detection.")

    # Check for invalid or unexpected values
    invalid_values = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for unexpected values in categorical columns
            unexpected = df[col].value_counts().index.tolist()
            invalid_values[col] = unexpected
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Check for negative values in numerical columns (if applicable)
            if (df[col] < 0).any():
                invalid_values[col] = "Contains negative values"
    results['invalid_values'] = invalid_values


    return results
# ===========================================
# Function for Initial EDA with Visualizations
def initial_eda(df):
    """
    Generate a Sweetviz EDA report for the entire DataFrame and display it in the notebook.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    """
    # Generate the Sweetviz report
    report = sv.analyze(df)
    
    # Display the report in the notebook
    report.show_notebook()

# ============================================================================================
# Function to display dataset statistics
def descriptive_statistics(df):
    """
    Generate descriptive summary for each column and plot frequency distribution for categorical columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    """
    # Loop through each column and generate descriptive summary
    for col in df.columns:
        print(f"\nDescriptive Summary for {col}:")
        if df[col].dtype in ['int64', 'float64']:  # Numerical columns
            print(df[col].describe())
        else:  # Categorical columns
            print(df[col].value_counts())

        # Plot frequency distribution for categorical columns
        if df[col].dtype in ['object', 'category']:
            # Check if the column has many unique values or long names
            if df[col].nunique() > 10 or any(len(str(x)) > 15 for x in df[col].unique()):
                # Horizontal bar chart
                plt.figure(figsize=(8, 6))
                sns.countplot(y=df[col], palette='viridis', order=df[col].value_counts().index)
                plt.title(f"Frequency of {col}")
                plt.xlabel("Count")
                plt.ylabel(col)
                plt.show()
            else:
                # Vertical bar chart
                plt.figure(figsize=(8, 6))
                sns.countplot(x=df[col], palette='viridis', order=df[col].value_counts().index)
                plt.title(f"Frequency of {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                plt.show()

# =====================================================================================
# Attrition Rate Analysis Function
def attrition_rate_analysis(df, groupby_cols, target='attrition'):
    """
    Calculate and visualize attrition rates by specified grouping columns.

    Parameters:
    df (pd.DataFrame): The dataset.
    groupby_cols (list): Columns to group by (e.g., demographics or job-related columns).
    target (str): The target variable (default is 'attrition').
    """
    # Calculate overall attrition rate
    overall_attrition = df[target].mean() * 100
    print(f"Overall Attrition Rate: {overall_attrition:.2f}%")

    # Calculate attrition rates by specified columns
    for col in groupby_cols:
        attrition_rate = df.groupby(col)[target].mean() * 100
        print(f"\nAttrition Rate by {col}:")
        print(attrition_rate)

        # Plot bar chart
        plt.figure(figsize=(8, 6))
        sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette='viridis')
        plt.title(f"Attrition Rate by {col}")
        plt.ylabel("Attrition Rate (%)")
        plt.xticks(rotation=45)
        plt.show()
# =====================================================================================
# Function for Correlation Analysis

def correlation_analysis(df, numerical_cols, categorical_cols, target):
    # Numerical correlation
    numerical_corr = df[numerical_cols + [target]].corr()
    print("Correlation Matrix for Numerical Variables:")
    print(numerical_corr)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix for Numerical Variables")
    plt.show()

    # Categorical correlations (Cramer's V)
    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df[target])
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        cramer_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))  # Fixed line
        print(f"Cramer's V for {col}: {cramer_v:.2f}")


# ==========================================================================================
# Function for Predictive Modeling with Multiple Models

def predictive_modeling(df, features, target):
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Select features and target
    X = df_encoded[features]
    y = df_encoded[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    # Evaluate models
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba.any() else None

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

    # Display results
    results_df = pd.DataFrame(results)
    print(results_df)

    # Plot comparison
    results_df.set_index("Model").plot(kind='bar', figsize=(12, 8), title="Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.show()

# =====================================================================
# Function for Cluster Analysis

def cluster_analysis(df, features):
    # Select features
    X = df[features]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Analyze clusters
    cluster_summary = df.groupby('cluster')[features].mean()
    print(cluster_summary)

    # Visualize clusters
    sns.pairplot(df, vars=features, hue='cluster', palette='viridis')
    plt.show()

# ========================================================================
# Function for Survival Analysis

def survival_analysis(df, time_col, event_col, predictors):
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df[[time_col, event_col] + predictors], duration_col=time_col, event_col=event_col)

    # Print summary
    print(cph.print_summary())

    # Plot survival curves
    cph.plot()














    