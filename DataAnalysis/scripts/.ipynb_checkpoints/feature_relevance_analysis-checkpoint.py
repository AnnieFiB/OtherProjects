
# feature_relevance_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def encode_dataframe(df, target_column):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    return df_encoded

def compute_random_forest_importance(df, target_column='churn', top_n=20, plot=True):
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

def compute_mutual_info(df, target_column='churn', top_n=20, plot=True):
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

def compute_combined_feature_importance(df, target_column='churn', top_n=20, plot=True):
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

def prepare_features(df, target_column='churn', drop_columns=None):
    import pandas as pd

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


def scale_and_split(df, target_column='churn', test_size=0.2, random_state=42):
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
        ('scaler', StandardScaler())
    ])

    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, pipeline


from imblearn.over_sampling import SMOTE

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
