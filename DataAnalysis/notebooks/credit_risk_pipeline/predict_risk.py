# credit_risk_pipeline/predict_risk.py (CLI usage)

from preprocessor import *

import pandas as pd
import joblib

class CreditRiskPredictor:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, raw_df: pd.DataFrame):
        processed = self.preprocessor.transform(raw_df)
        pred = self.model.predict(processed)
        return pred

    def predict_proba(self, raw_df: pd.DataFrame):
        processed = self.preprocessor.transform(raw_df)
        proba = self.model.predict_proba(processed)
        return proba[:, 1]