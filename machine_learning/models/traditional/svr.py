from .base_model import BaseMLModel
from sklearn.svm import SVR
import pandas as pd
import numpy as np

class SVRModel(BaseMLModel):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = SVR(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
