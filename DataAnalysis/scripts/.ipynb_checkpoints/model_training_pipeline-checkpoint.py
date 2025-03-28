
# model_training_pipeline.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay, auc
import joblib
from sklearn.model_selection import GridSearchCV
import shap
from sklearn.preprocessing import StandardScaler



def train_and_evaluate_models(X, y, cv_splits=5):
    """
    Train models using cross validation scores to avoid overfitting
    """    
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
      }

    results = {}
    trained_models = {}

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Evaluating: {name}")

        y_pred = cross_val_predict(model, X, y, cv=skf)
        if hasattr(model, "predict_proba"):
            y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
        else:
            y_proba = None

        report = classification_report(y, y_pred, output_dict=True)
        auc_score = roc_auc_score(y, y_proba) if y_proba is not None else None
        cm = confusion_matrix(y, y_pred).tolist()

        results[name] = {
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": auc_score
        }

        # Fit model on entire data for final model use
        model.fit(X, y)
        trained_models[name] = model

    return results, trained_models

def results_to_dataframe(results_dict):
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



def tune_and_select_best_model(X_train, y_train, models_and_params, save_path="best_model.pkl"):
    '''
    Tune given models using GridSearchCV and select the one with the highest Recall (1),
    while reporting F1-score for transparency. Skips models that raise compatibility errors.

    Parameters:
    - X_train, y_train: Training data
    - models_and_params: dict of model name -> (model instance, param_grid)
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

    for name, (model, param_grid) in models_and_params.items():
        try:
            print(f"🔍 Tuning {name}...")
            grid = GridSearchCV(model, param_grid, scoring='recall', cv=5, n_jobs=1, verbose=0, refit=True)
            grid.fit(X_train, y_train)

            recall_score = grid.best_score_
            f1_grid = GridSearchCV(model, param_grid, scoring='f1', cv=5, n_jobs=1, verbose=0)
            f1_grid.fit(X_train, y_train)
            f1_score = f1_grid.best_score_

            print(f" {name} Best Recall: {recall_score:.4f} | F1: {f1_score:.4f} | Params: {grid.best_params_}")

            if recall_score > best_recall:
                best_recall = recall_score
                best_model = grid.best_estimator_
                best_name = name
                best_scores = {"recall": recall_score, "f1": f1_score}

        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

    if best_model:
        print(f" Selected Model: {best_name} | Recall: {best_scores['recall']:.4f} | F1: {best_scores['f1']:.4f}")
        joblib.dump(best_model, save_path)
        print(f" Best model saved to {save_path}")
    else:
        print("No valid model could be tuned.")

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
