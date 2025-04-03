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
from sklearn.dummy import DummyClassifier
from scipy.stats import chi2_contingency

# -----------------------------
# Data Processing
# -----------------------------
def detect_columns(df):
    st.info("🔍 Detecting column types for configuration")
    config = {'num_cols': [], 'binary_cols': [], 'multi_cat_cols': [], 'date_cols': [], 'irrelevant_cols': []}
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
    st.info("🧼 Cleaning data: handling dates, missing values, and duplicates")
    df = df.drop(columns=config['irrelevant_cols'])
    if config['date_cols']:
        for col in config['date_cols']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if missing_strategy == 'fill':
        df = df.fillna({col: df[col].median() if col in config['num_cols'] else df[col].mode()[0] for col in df.columns})
    elif missing_strategy == 'drop':
        df = df.dropna().reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

# -----------------------------
# Feature Engineering & Modeling
# -----------------------------
def prepare_features(df, target_column='churn', drop_columns=None):
    st.info("🧹 Preprocessing: Encoding target column and categorical features")
    df = df.copy()
    if df[target_column].dtype == 'object':
        st.info(f"ℹ️ Converting target '{target_column}' from Yes/No to 1/0")
        df[target_column] = df[target_column].str.lower().map({'no': 0, 'yes': 1})
    df = df.dropna(subset=[target_column])
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    feature_cols = df.columns.drop(target_column)
    st.info("🔧 Encoding categorical variables")
    for col in df[feature_cols].select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()
        df[col] = df[col].astype('category').cat.codes
    X = df[feature_cols]
    y = df[target_column]
    return X, y

def scale_and_split(df, target_column='churn', test_size=0.2, random_state=42):
    X, y = prepare_features(df, target_column=target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    pipeline = Pipeline([('scaler', StandardScaler())])
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, pipeline

def train_and_evaluate_models(X, y, cv_splits=10):
    st.info("⚙️ Training and evaluating baseline models")
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "SVM": SVC(probability=True, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    results, trained_models = {}, {}
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=skf)
        y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1] if hasattr(model, "predict_proba") else None
        report = classification_report(y, y_pred, output_dict=True)
        auc_score = roc_auc_score(y, y_proba) if y_proba is not None else None
        cm = confusion_matrix(y, y_pred).tolist()
        results[name] = {"classification_report": report, "confusion_matrix": cm, "roc_auc": auc_score}
        model.fit(X, y)
        trained_models[name] = model
    return results, trained_models

def results_to_dataframe(results_dict):
    rows = []
    for model_name, metrics in results_dict.items():
        clf_report = metrics['classification_report']
        row = {
            'Model': model_name,
            'ROC AUC': metrics['roc_auc'],
            'Accuracy': clf_report['accuracy'],
            'Precision (1)': clf_report['1']['precision'],
            'Recall (1)': clf_report['1']['recall'],
            'F1-score (1)': clf_report['1']['f1-score']
        }
        rows.append(row)
    return pd.DataFrame(rows)

def tune_and_select_best_model(X_train, y_train, models_and_params, save_path="best_model.pkl"):
    st.info("🛠 Fine-tuning models and selecting the best one")
    best_model, best_recall, best_name, best_scores = None, 0, None, {}
    for name, (model, param_grid) in models_and_params.items():
        try:
            grid = GridSearchCV(model, param_grid, scoring='recall', cv=5, refit=True)
            grid.fit(X_train, y_train)
            recall_score = grid.best_score_
            f1_grid = GridSearchCV(model, param_grid, scoring='f1', cv=5)
            f1_grid.fit(X_train, y_train)
            f1_score = f1_grid.best_score_
            if recall_score > best_recall:
                best_model = grid.best_estimator_
                best_recall = recall_score
                best_name = name
                best_scores = {"recall": recall_score, "f1": f1_score}
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
    if best_model:
        joblib.dump(best_model, save_path)
    return best_model, best_name, best_scores

# -----------------------------
# EDA Functions
# -----------------------------
def attrition_rate_analysis(df, groupby_cols, target='attrition'):
    overall_attrition = df[target].mean() * 100
    print(f"Overall Attrition Rate: {overall_attrition:.2f}%")
    for col in groupby_cols:
        attrition_rate = df.groupby(col)[target].mean() * 100
        print(f"\nAttrition Rate by {col}:")
        print(attrition_rate)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette='viridis')
        plt.title(f"Attrition Rate by {col}")
        plt.ylabel("Attrition Rate (%)")
        plt.xticks(rotation=45)
        plt.show()

def correlation_analysis(df, numerical_cols, categorical_cols, target):
    numerical_corr = df[numerical_cols + [target]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix for Numerical Variables")
    plt.show()
    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df[target])
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        cramer_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        print(f"Cramer's V for {col}: {cramer_v:.2f}")
