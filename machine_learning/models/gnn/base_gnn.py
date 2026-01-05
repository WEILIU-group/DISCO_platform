from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseGNNModel(ABC):
    """
    Abstract base class for Graph Neural Network models.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.model = None
        self.device = 'cpu' # or 'cuda' if available

    @abstractmethod
    def train(self, dataset: List[Dict], val_dataset: List[Dict] = None):
        """
        Train the model.
        
        Args:
            dataset: List of graph data dicts.
            val_dataset: Validation dataset.
        """
        pass

    @abstractmethod
    def predict(self, dataset: List[Dict]) -> np.ndarray:
        """
        Predict using the trained model.
        """
        pass
