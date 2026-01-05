from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseMLModel(ABC):
    """
    Abstract base class for traditional machine learning models.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.model = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the trained model."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        return self.params
