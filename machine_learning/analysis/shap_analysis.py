import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Any, Optional

class ShapAnalyzer:
    """
    SHAP analysis for machine learning models.
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained model object.
            X_train: Training data (used as background for some explainers).
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        
        # Determine explainer type
        # This is a simplified logic; might need adjustment based on specific model libraries
        model_name = str(type(model)).lower()
        
        try:
            if "xgboost" in model_name or "randomforest" in model_name or "tree" in model_name:
                self.explainer = shap.TreeExplainer(model)
            elif "linear" in model_name:
                self.explainer = shap.LinearExplainer(model, X_train)
            else:
                # KernelExplainer is generic but slow
                # Using a subset of X_train as background to speed up
                background = shap.kmeans(X_train, 10) if len(X_train) > 100 else X_train
                self.explainer = shap.KernelExplainer(model.predict, background)
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")

    def calculate_shap_values(self, X: pd.DataFrame):
        """
        Calculate SHAP values for the given data.
        """
        if self.explainer is None:
            return None
            
        try:
            shap_values = self.explainer.shap_values(X)
            # For some models (like RF classifier), shap_values is a list. 
            # Assuming regression here, so it should be a single array.
            # If it's a list (e.g. multi-output), take the first one or handle accordingly.
            if isinstance(shap_values, list):
                return shap_values[0]
            return shap_values
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None

    def plot_summary(self, shap_values, X: pd.DataFrame, save_path: Optional[str] = None):
        """
        Generate SHAP summary plot.
        """
        if shap_values is None:
            return
            
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
