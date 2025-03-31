# hrattrition.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.dummy import DummyClassifier
from scipy.stats import chi2_contingency, f_oneway
from imblearn.over_sampling import SMOTE

# -----------------------------
# Data Processing
# -----------------------------
def detect_columns(df):
    """Automatically detect column types and return configuration"""
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
            else:
                config['irrelevant_cols'].append(col)
    
    return config

def clean_data(df, config, missing_strategy='fill'):
    """Clean and preprocess data with proper type conversions"""
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    # Convert target column
    if 'attrition' in df.columns:
        df['attrition'] = df['attrition'].str.lower().map({'yes': 1, 'no': 0})
    
    # Process configuration
    lower_config = {k.lower(): [col.lower() for col in v] for k, v in config.items()}
    
    # Drop irrelevant columns
    irrelevant_cols = [col for col in lower_config.get('irrelevant_cols', []) if col in df.columns]
    df = df.drop(columns=irrelevant_cols)
    
    # Handle date columns
    for col in lower_config.get('date_cols', []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Handle missing values
    if missing_strategy == 'fill':
        num_cols = lower_config.get('num_cols', [])
        for col in df.columns:
            if col in num_cols:
                df[col] = df[col].fillna(df[col].median())  # Changed from inplace=True
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else None
                df[col] = df[col].fillna(mode_val)  # Changed from inplace=True
    elif missing_strategy == 'drop':
        df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df

# -----------------------------
# Statistical Analysis
# -----------------------------
def chi_square_test_batch(df, cat_columns, target_column='attrition'):
    """Batch process chi-square tests for categorical features"""
    results = []
    for col in cat_columns:
        try:
            contingency_table = pd.crosstab(df[col], df[target_column])
            chi2, p, dof, _ = chi2_contingency(contingency_table)
            results.append({
                'feature': col,
                'p_value': p,
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

def anova_test_numerical_features(df, numeric_columns, target_column='attrition'):
    """Perform ANOVA tests for numerical features without plotting"""
    results = []
    
    if df[target_column].dtype != 'int64' and df[target_column].dtype != 'float64':
        df[target_column] = df[target_column].map({'no': 0, 'yes': 1})

    for col in numeric_columns:
        if col not in df.columns or df[col].nunique() <= 1:
            continue
            
        try:
            group1 = df[df[target_column] == 0][col].dropna()
            group2 = df[df[target_column] == 1][col].dropna()
            f_stat, p_value = f_oneway(group1, group2)
            results.append({
                'feature': col,
                'p_value': round(p_value, 4),
                'significant': p_value < 0.05
            })
        except Exception as e:
            results.append({
                'feature': col,
                'p_value': None,
                'significant': False,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

# -----------------------------
# Visualization
# -----------------------------
def plot_significant_categorical_proportions(df, features, target_column='attrition'):
    """Plot proportions for significant categorical features"""
    figures = []
    for col in features:
        fig, ax = plt.subplots(figsize=(10, 6))
        prop_df = pd.crosstab(df[col], df[target_column], normalize='index')
        prop_df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Attrition Proportions by {col}')
        figures.append(fig)
    return figures

def correlation_analysis(df, numerical_cols):
    """Generate correlation matrix visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    return fig

# -----------------------------
# Modeling
# -----------------------------
def prepare_features(df, target_column='attrition', drop_columns=None):
    """Prepare features for modeling"""
    df = df.copy()
    if drop_columns:
        df = df.drop(columns=[col.lower() for col in drop_columns if col.lower() in df.columns])
    
    # Convert categorical features
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def scale_and_split(df, target_column='attrition', test_size=0.2):
    """Create train/test split with scaling"""
    X, y = prepare_features(df, target_column)
    pipeline = Pipeline([('scaler', StandardScaler())])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    return X_train, X_test, y_train, y_test, pipeline

def balance_classes_smote(X, y):
    """Balance classes using SMOTE"""
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

def train_and_evaluate_models(X, y, cv=5):
    """Train and evaluate multiple models with proper report handling"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    
    results = {}
    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=cv)
        y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1] if hasattr(model, "predict_proba") else None
        
        # Generate classification report with explicit labels
        report = classification_report(y, y_pred, output_dict=True, labels=[0,1])
        
        results[name] = {
            'roc_auc': roc_auc_score(y, y_proba) if y_proba is not None else None,
            'accuracy': report['accuracy'],
            'precision_0': report['0']['precision'],
            'recall_0': report['0']['recall'],
            'f1_0': report['0']['f1-score'],
            'precision_1': report['1']['precision'],
            'recall_1': report['1']['recall'],
            'f1_1': report['1']['f1-score']
        }
    return results, models


def tune_and_select_best_model(X, y, models_and_params):
    """Hyperparameter tuning with GridSearchCV returning model, name, and metrics"""
    best_score = -1
    best_model = None
    best_name = ""
    best_scores = {}
    
    for name, (model, params) in models_and_params.items():
        with st.spinner(f"⚙️ Tuning {name}..."):
            # Use both recall and F1 in scoring
            grid = GridSearchCV(model, params, 
                              scoring={'recall': 'recall', 'f1': 'f1'},
                              refit='recall',  # Optimize for recall but track F1
                              cv=5,
                              n_jobs=-1)
            grid.fit(X, y)
            
            # Get best scores
            current_recall = grid.cv_results_['mean_test_recall'][grid.best_index_]
            current_f1 = grid.cv_results_['mean_test_f1'][grid.best_index_]
            
            if current_recall > best_score:
                best_score = current_recall
                best_model = grid.best_estimator_
                best_name = name
                best_scores = {
                    'recall': current_recall,
                    'f1': current_f1,
                    'best_params': grid.best_params_
                }
    
    return best_model, best_name, best_scores

def results_to_dataframe(results_dict):
    rows = []
    for model_name, metrics in results_dict.items():
        row = {
            'Model': model_name,
            'ROC AUC': metrics['roc_auc'],
            'Accuracy': metrics['accuracy'],
            'Precision (0)': metrics['precision_0'],
            'Recall (0)': metrics['recall_0'],
            'F1 (0)': metrics['f1_0'],
            'Precision (1)': metrics['precision_1'],
            'Recall (1)': metrics['recall_1'],
            'F1 (1)': metrics['f1_1']
        }
        rows.append(row)
    return pd.DataFrame(rows)

def attrition_rate_analysis(df, groupby_cols, target='attrition'):
    overall_attrition = df[target].mean() * 100
    fig = plt.figure(figsize=(10, 6))
    
    for col in groupby_cols:
        attrition_rate = df.groupby(col)[target].mean() * 100
        plt.figure(figsize=(8, 6))
        sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette='viridis')
        plt.title(f"Attrition Rate by {col}")
        plt.ylabel("Attrition Rate (%)")
        plt.xticks(rotation=45)
    
    return fig  # Return figure instead of showing


def encode_dataframe(df, target_column):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    return df_encoded

def compute_random_forest_importance(df, target_column='attrition', top_n=20, plot=True):
    if df[target_column].dtype == 'object':
        if set(df[target_column].unique()) == {'yes', 'no'}:
            df[target_column] = df[target_column].map({'no': 0, 'yes': 1})
        else:
            raise ValueError(f"Target column '{target_column}' must be binary and convertible to 0/1.")

    df_encoded = encode_dataframe(df, target_column)
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_rf': model.feature_importances_
    }).sort_values(by='importance_rf', ascending=False)

    if plot:
        top_features = importance_df.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['feature'], top_features['importance_rf'], color='teal')
        plt.gca().invert_yaxis()
        plt.title('Top Feature Importances (Random Forest)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    return importance_df

def compute_mutual_info(df, target_column='attrition', top_n=20, plot=True):
    if df[target_column].dtype == 'object':
        if set(df[target_column].unique()) == {'yes', 'no'}:
            df[target_column] = df[target_column].map({'no': 0, 'yes': 1})
        else:
            raise ValueError(f"Target column '{target_column}' must be binary and convertible to 0/1.")

    df_encoded = encode_dataframe(df, target_column)
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]

    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_df = pd.DataFrame({'feature': X.columns, 'importance_mi': mi_scores})
    mi_df = mi_df.sort_values(by='importance_mi', ascending=False)

    if plot:
        top_features = mi_df.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['feature'], top_features['importance_mi'], color='coral')
        plt.gca().invert_yaxis()
        plt.title('Top Feature Importances (Mutual Information)')
        plt.xlabel('Mutual Information Score')
        plt.tight_layout()
        plt.show()

    return mi_df

def compute_combined_feature_importance(df, target_column='attrition', top_n=20, plot=True):
    rf_df = compute_random_forest_importance(df.copy(), target_column, top_n=top_n, plot=False)
    mi_df = compute_mutual_info(df.copy(), target_column, top_n=top_n, plot=False)

    merged_df = pd.merge(rf_df, mi_df, on='feature', how='outer').fillna(0)
    merged_df = merged_df.sort_values(by='importance_rf', ascending=False)

    if plot:
        merged_top = merged_df.head(top_n).set_index('feature')
        merged_top.plot(kind='barh', figsize=(10, 6), color=['teal', 'coral'])
        plt.gca().invert_yaxis()
        plt.title('Top Feature Importances: RF vs MI')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    return merged_df