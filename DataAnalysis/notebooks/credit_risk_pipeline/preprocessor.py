# credit_risk_pipeline/pipeline/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
import joblib

class PreprocessingPipeline:
    def __init__(self):
        self.ohe = None
        self.oe = None
        self.scaler = None
        self.feature_names_ = None
        self.binary_cols = None
        self.categorical_cols = None
        self.numeric_cols = None

    def fit(self, df):
        df = self._clean(df)

        self.binary_cols = ['gender', 'email']
        self.categorical_cols = ['employment_status']
        self.numeric_cols = ['income', 'age', 'employment_length']

        self.oe = OrdinalEncoder()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = MinMaxScaler()

        self.oe.fit(df[self.binary_cols])
        self.ohe.fit(df[self.categorical_cols])
        self.scaler.fit(df[self.numeric_cols])

        bin_cols = self.binary_cols
        cat_cols = list(self.ohe.get_feature_names_out(self.categorical_cols))
        num_cols = self.numeric_cols

        self.feature_names_ = list(pd.Index(bin_cols + cat_cols + num_cols).drop_duplicates())


    def transform(self, df):
        df = self._clean(df)

        bin_encoded = self.oe.transform(df[self.binary_cols])
        bin_df = pd.DataFrame(bin_encoded, columns=self.binary_cols, index=df.index)

        cat_encoded = self.ohe.transform(df[self.categorical_cols])
        cat_df = pd.DataFrame(cat_encoded, columns=self.ohe.get_feature_names_out(self.categorical_cols), index=df.index)

        num_scaled = self.scaler.transform(df[self.numeric_cols])
        num_df = pd.DataFrame(num_scaled, columns=self.numeric_cols, index=df.index)

        final_df = pd.concat([bin_df, cat_df, num_df], axis=1)

        # Validate feature shape
        if self.feature_names_ and sorted(final_df.columns.tolist()) != sorted(self.feature_names_):
            missing = set(self.feature_names_) - set(final_df.columns)
            raise ValueError(f"Feature mismatch after transform.\nExpected: {self.feature_names_}\nGot: {list(final_df.columns)}\nMissing: {missing}")

        return final_df.reindex(columns=self.feature_names_, fill_value=0)


    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def _clean(self, df):
        df = df.copy()

        rename_cols = {
            'CODE_GENDER': 'gender',
            'FLAG_EMAIL': 'email',
            'AMT_INCOME_TOTAL': 'income',
            'DAYS_BIRTH': 'age',
            'DAYS_EMPLOYED': 'employment_length',
            'NAME_INCOME_TYPE': 'employment_status',
        }
        df = df.rename(columns=rename_cols)

        df['age'] = ((df['age']).astype('Int64') / -365).astype('float')
        df['employment_length'] = df['employment_length'].apply(
            lambda x: 0 if x > 0 else int(abs(x) / 365))

        df['employment_status'] = df['employment_status'].astype(str).str.strip().str.lower()
        return df
