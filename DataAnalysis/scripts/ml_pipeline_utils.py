
## feature_relevance_analysis.py and model_training_pipeline.py


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler,  OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import boxcox
import joblib
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
    
# Set global seeds
np.random.seed(42)
random.seed(42)

# =========================================
# Data Preprocessing Functions
# =========================================

def transform_skewed_columns(df, columns, method='cbrt'):
    """
    Apply skewness-reducing transformation to specified columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - columns (list): List of column names to transform
    - method (str): 'cbrt' or 'log'

    Returns:
    - pd.DataFrame: Transformed DataFrame
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found. Skipping.")
            continue

        try:
            if method == 'cbrt':
                df[col] = np.cbrt(df[col]).round(1)
            elif method == 'log':
                df[col] = np.log1p(df[col]).round(1)
            else:
                print(f"⚠️ Method '{method}' not recognized. Use 'cbrt' or 'log'.")
        except Exception as e:
            print(f"⚠️ Could not transform column '{col}': {e}")

    return df

def encode_dataframe(df, target_column):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    return df_encoded

def encode_columns(df, target_col=None):
    """
    Automatically encode categorical features:
    - One-hot encode multi-category columns
    - Ordinal encode binary columns
    - Encode target column if it's not numeric

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - target_col (str): Optional target column to preserve as single int

    Returns:
    - pd.DataFrame: Encoded DataFrame
    """
    df = df.copy()
    
    # Separate target column
    if target_col and target_col in df.columns:
        target_series = df[target_col].copy()
        if not pd.api.types.is_numeric_dtype(target_series):
            print(f"ℹ️ Encoding target column '{target_col}' to numeric.")
            target_series = LabelEncoder().fit_transform(target_series.astype(str))
        df.drop(columns=[target_col], inplace=True)
    else:
        target_series = None

    # Auto-detect categorical columns (exclude target)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols += [
    col for col in df.columns 
    if df[col].nunique() > 2 and not is_numeric_dtype(df[col])
    ]
    cat_cols = list(set(cat_cols))

    # Categorize them into binary or multi-class
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    multi_class_cols = [col for col in cat_cols if df[col].nunique() > 2]

    # Ordinal encode binary cols
    if binary_cols:
        oe = OrdinalEncoder()
        df[binary_cols] = oe.fit_transform(df[binary_cols])

    # One-hot encode multi-class cols
    if multi_class_cols:
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        encoded = ohe.fit_transform(df[multi_class_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(multi_class_cols), index=df.index)
        df.drop(columns=multi_class_cols, inplace=True)
        df = pd.concat([df, encoded_df], axis=1)

    # Reattach target column if it existed
    if target_col:
        df[target_col] = target_series

    return df

def apply_min_max_scaling(df, columns=None, target_col=None):
    """
    Apply Min-Max scaling to numeric columns, excluding the target column if specified.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - columns (list or None): Columns to scale; if None, scales all numeric columns except target
    - target_col (str or None): Column name to exclude from scaling (typically the target variable)

    Returns:
    - df_scaled (pd.DataFrame): DataFrame with scaled features
    """
    df_scaled = df.copy()
    scaler = MinMaxScaler()

    if columns is None:
        columns = df_scaled.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)

    df_scaled[columns] = scaler.fit_transform(df_scaled[columns]).round(1)
    return df_scaled

# =========================================
# Feature Importance Functions
# =========================================

def compute_random_forest_importance(df, target_column='y', top_n=20, plot=True):
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

def compute_mutual_info(df, target_column='y', top_n=20, plot=True):
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

def compute_combined_feature_importance(df, target_column='y', top_n=20, plot=True):
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

def prepare_features(df, target_column='y', drop_columns=None):
  
    df = df.copy()

    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].str.lower().map({'no': 0, 'yes': 1})

    df = df.dropna(subset=[target_column])

    if drop_columns:
        drop_columns = [col for col in drop_columns if col in df.columns]
        df.drop(columns=drop_columns, inplace=True)

    df.dropna(inplace=True)

    feature_cols = df.columns.drop(target_column)

    for col in df[feature_cols].select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()
        df[col] = df[col].astype('category').cat.codes

    X = df[feature_cols]
    y = df[target_column]

    return X, y

# =========================================
# ML Pipeline Functions
# =========================================
def scale_and_split(df, target_column='y', test_size=0.2, random_state=42):
    '''
    Scales and splits features returned by prepare_features.

    Assumes drop_columns (if needed) were handled inside prepare_features.

    Parameters:
    - df (pd.DataFrame): Input dataset
    - target_column (str): Target variable
    - test_size (float): Proportion of test data
    - random_state (int): Random seed

    Returns:
    - X_train_scaled, X_test_scaled, y_train, y_test, pipeline
    '''
    X, y = prepare_features(df, target_column=target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()) # or StandardScaler() based on needs
    ])

    X_train_scaled = pipeline.fit_transform(X_train).round(1)
    X_test_scaled = pipeline.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, pipeline

def scale_train_test(X_train, X_test):
    """
    Scales X_train and X_test using a pipeline with StandardScaler.

    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Training features
    - X_test (pd.DataFrame or np.ndarray): Testing features

    Returns:
    - X_train_scaled (np.ndarray): Scaled training features
    - X_test_scaled (np.ndarray): Scaled test features
    - pipeline (Pipeline): Fitted sklearn pipeline with StandardScaler
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    return X_train_scaled, X_test_scaled, pipeline

def balance_classes_smote(X, y, random_state=42):
    '''
    Applies SMOTE to balance the dataset.

    Parameters:
    - X (array or DataFrame): Feature matrix
    - y (array or Series): Target labels

    Returns:
    - X_resampled, y_resampled: Balanced feature matrix and labels
    '''
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def train_and_evaluate_models(X, y, selected_models=None, cv_splits=10):
    """
    Train selected models using cross-validation and return performance metrics.
    
    Parameters:
    - X, y: Features and target
    - selected_models (list of str): List of model names to train. If None, all models will be used.
    - cv_splits (int): Number of StratifiedKFold splits

    Returns:
    - results (dict): Model evaluation metrics
    - trained_models (dict): Fitted models
    """

    all_models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, class_weight="balanced"),
        "LightGBM": LGBMClassifier(class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss", scale_pos_weight=5),
        "CatBoost": CatBoostClassifier(verbose=0, class_weights=[1, 5]),  # Adjust weight if needed,
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
    }

    # Filter models if user specifies a subset
    models_to_run = {k: v for k, v in all_models.items() if (selected_models is None or k in selected_models)}

    results = {}
    trained_models = {}
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for name, model in models_to_run.items():
        print(f"Evaluating: {name}")

        y_pred = cross_val_predict(model, X, y, cv=skf)
        y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1] if hasattr(model, "predict_proba") else None

        report = classification_report(y, y_pred, output_dict=True)
        auc_score = roc_auc_score(y, y_proba) if y_proba is not None else None
        cm = confusion_matrix(y, y_pred).tolist()

        results[name] = {
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": auc_score
        }

        # Train final model on full data
        model.fit(X, y)
        trained_models[name] = model

    return results, trained_models

def results_to_dataframe1(results_dict):
    '''
    Convert evaluation metrics from model dictionary to a pandas DataFrame.

    Parameters:
    - results_dict (dict): Output from train_and_evaluate_models

    Returns:
    - DataFrame with summary metrics for comparison
    '''
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

def results_to_dataframe(results_dict):
    """
    Convert evaluation metrics from model dictionary to a pandas DataFrame.
    Handles both '1' and '1.0' (str/float) as positive class labels.
    """
    rows = []

    for model_name, metrics in results_dict.items():
        clf_report = metrics['classification_report']
        # Normalize keys as strings for consistency
        keys = list(clf_report.keys())
        pos_class_key = None

        # Find the positive class key among '1', 1, '1.0', 1.0
        for k in ['1', 1, '1.0', 1.0]:
            if str(k) in keys:
                pos_class_key = str(k)
                break
        def safe_round(val):
            return round(val, 2) if val is not None else None
        
        row = {
            'Model': model_name,
            'ROC AUC': safe_round(metrics.get('roc_auc', None)),
            'Accuracy': safe_round(clf_report.get('accuracy', None)),
            'Precision (1)': safe_round(clf_report.get(pos_class_key, {}).get('precision', None)),
            'Recall (1)': safe_round(clf_report.get(pos_class_key, {}).get('recall', None)),
            'F1-score (1)': safe_round(clf_report.get(pos_class_key, {}).get('f1-score', None))
        }
        rows.append(row)

    return pd.DataFrame(rows)

def plot_model_evaluations(models, X_test, y_test):
    '''
    Plots a shared ROC curve for all models and individual confusion matrices.

    Parameters:
    - models (dict): Dictionary of model_name: trained_model
    - X_test (array): Test features
    - y_test (array): True labels
    '''

    # Shared ROC Curve
    plt.figure(figsize=(10, 6))
    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.title("ROC Curve - All Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Confusion Matrices
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(6, 4 * n_models))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f"{model_name} - Confusion Matrix")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

def load_models_and_params():
    """
    Loads a dictionary of models and their parameter distributions for RandomizedSearchCV.

    Returns:
    - models_and_params (dict): {model_name: (model_instance, param_distributions)}
    """
    models_and_params = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight='balanced'),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight='balanced'),
            {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        ),
        "Naive Bayes": (
            GaussianNB(),
            {}  # No useful hyperparameters to tune
        ),
        "Neural Network": (
            MLPClassifier(max_iter=1000),
            {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        ),
        "SVM": (
            SVC(probability=True, class_weight='balanced'),
            {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        ),
        "LightGBM": (
            LGBMClassifier(class_weight='balanced'),
            {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [-1, 10, 20]
            }
        ),
        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=5),
            {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 6, 10]
            }
        ),
        "CatBoost": (
            CatBoostClassifier(verbose=0, class_weights=[1, 5]),
            {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 10]
            }
        )
    }

    return models_and_params

def tune_and_select_best_model(X_train, y_train, models_and_params, model_to_tune=None, save_path="best_model.joblib"):
    '''
    Tune a specified model using GridSearchCV and select the best estimator based on Recall.
    
    Parameters:
    - X_train, y_train: Training data
    - models_and_params: dict of model name -> (model instance, param_grid)
    - model_to_tune: (str) Optional. Name of the model to tune. If None, tunes all and selects best.
    - save_path: path to save the best model
    
    Returns:
    - best_estimator: trained best model
    - best_name: name of the best model
    - best_scores: dict with best recall and f1
    '''

    best_model = None
    best_recall = 0
    best_name = None
    best_scores = {}

    models_to_run = models_and_params if model_to_tune is None else {
        model_to_tune: models_and_params[model_to_tune]
    }

    for name, (model, param_grid) in models_to_run.items():
        try:
            print(f"🔍 Tuning {name}...")
            grid = GridSearchCV(model, param_grid, scoring='recall', cv=5, n_jobs=-1, verbose=0, refit=True)
            grid.fit(X_train, y_train)

            recall_best = grid.best_score_

            # Recalculate F1 using the best estimator (not a separate grid search)
            f1_best = f1_score(y_train, grid.predict(X_train))

            print(f" ✅ {name} Best Recall: {recall_best:.4f} | F1: {f1_best:.4f} | Params: {grid.best_params_}")

            if recall_best > best_recall:
                best_recall = recall_best
                best_model = grid.best_estimator_
                best_name = name
                best_scores = {"recall": recall_best, "f1": f1_best}

        except Exception as e:
            print(f"❌ Skipping {name} due to error: {e}")

    if best_model:
        print(f"\n🏆 Selected Model: {best_name} | Recall: {best_scores['recall']:.4f} | F1: {best_scores['f1']:.4f}")
        joblib.dump(best_model, save_path)
        print(f"💾 Best model saved to: {save_path}")
    else:
        print("⚠️ No valid model could be tuned.")

    return best_model, best_name, best_scores

def explain_model_with_shap(model_path, X_sample, feature_names=None, max_display=10):
    '''
    Generates SHAP explanations for a saved model using a sample of features.

    Parameters:
    - model_path (str): Path to the saved model .pkl file
    - X_sample (DataFrame): Sample feature set (preprocessed)
    - feature_names (list): Optional list of feature names
    - max_display (int): Max number of features to display in summary plot
    '''

    # Load model
    model = joblib.load(model_path)

    # Use TreeExplainer for tree-based models, KernelExplainer for others
    if "tree" in str(type(model)).lower():
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_sample)

    # Compute SHAP values
    shap_values = explainer(X_sample)

    # Plot summary
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=max_display)
