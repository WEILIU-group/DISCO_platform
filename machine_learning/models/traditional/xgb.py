from .base_model import BaseMLModel
import xgboost as xgb
import pandas as pd
import numpy as np

class XGBModel(BaseMLModel):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
