from typing import Dict, Any, Optional
import os
import json

class MLConfigManager:
    """
    Configuration Manager for Machine Learning Module.
    """
    
    DEFAULT_CONFIG = {
        "models": {
            "traditional": {
                "svr": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"},
                "gpr": {"alpha": 1e-10},
                "xgb": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
                "rf": {"n_estimators": 100, "max_depth": None},
                "mlp": {"hidden_layer_sizes": [100], "activation": "relu", "solver": "adam"}
            },
            "gnn": {
                "schnet": {"n_atom_basis": 128, "n_filters": 128, "n_interactions": 3},
                "dimenet_pp": {"n_blocks": 4, "n_bilinear": 8, "n_spherical": 7, "n_radial": 6}
            }
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "n_trials": 100,  # Default number of Optuna trials
            "n_jobs": 1       # Default number of parallel jobs for CV
        },
        "search_spaces": {
            "traditional": {
                "xgb": {
                    "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 1},
                    "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
                    "max_depth": {"type": "int", "low": 3, "high": 12, "step": 1},
                    "min_child_weight": {"type": "int", "low": 1, "high": 10},
                    "gamma": {"type": "float", "low": 0.0, "high": 5.0},
                    "subsample": {"type": "categorical", "choices": [0.5, 0.6, 0.8, 1.0]},
                    "colsample_bytree": {"type": "categorical", "choices": [0.5, 0.6, 0.8, 1.0]},
                    "colsample_bylevel": {"type": "categorical", "choices": [0.5, 0.6, 0.8, 1.0]},
                    "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                    "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True}
                },
                "svr": {
                    "C": {"type": "float", "low": 1e-2, "high": 1e2, "log": True},
                    "epsilon": {"type": "float", "low": 1e-3, "high": 1e0, "log": True},
                    "tol": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                    "kernel": {"type": "categorical", "choices": ["linear", "rbf", "poly"]},
                    "degree": {"type": "int", "low": 2, "high": 5, "condition": {"kernel": "poly"}},
                    "coef0": {"type": "float", "low": 0.0, "high": 1.0, "condition": {"kernel": "poly"}},
                    "gamma": {"type": "categorical", "choices": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0], "condition": {"kernel": ["rbf", "poly"]}}
                },
                "mlp": {
                    "hidden_layer_sizes": {"type": "categorical", "choices": [(64, 32), (128, 64), (256, 128), (128, 64, 32)]},
                    "activation": {"type": "categorical", "choices": ["relu", "tanh"]},
                    "solver": {"type": "categorical", "choices": ["adam"]},
                    "alpha": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                    "learning_rate_init": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}
                },
                "rf": {
                    "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 10},
                    "max_depth": {"type": "int", "low": 3, "high": 20, "step": 1},
                    "min_samples_split": {"type": "int", "low": 2, "high": 10},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 10}
                },
                "gpr": {
                    "alpha": {"type": "float", "low": 1e-10, "high": 1e-2, "log": True}
                }
            }
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, path: str):
        with open(path, 'r') as f:
            user_config = json.load(f)
            # Deep update could be implemented here, for now simple update
            self.config.update(user_config)

    def get_model_params(self, model_type: str, model_name: str) -> Dict[str, Any]:
        return self.config.get("models", {}).get(model_type, {}).get(model_name, {})

    def get_training_params(self) -> Dict[str, Any]:
        return self.config.get("training", {})
