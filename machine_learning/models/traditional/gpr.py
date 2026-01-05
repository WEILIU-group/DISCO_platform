from .base_model import BaseMLModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
import numpy as np

class GPRModel(BaseMLModel):
    def __init__(self, params=None):
        super().__init__(params)
        # Default kernel if not specified, though usually passed in params or constructed here
        # For simplicity, we let user pass kernel in params or use default
        if 'kernel' not in self.params:
            self.params['kernel'] = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        
        self.model = GaussianProcessRegressor(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
