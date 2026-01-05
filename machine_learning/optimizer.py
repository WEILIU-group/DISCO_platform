import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
from typing import Dict, Any, Optional, Callable
import joblib
import os

class HyperparameterOptimizer:
    """
    Handles hyperparameter optimization using Optuna.
    """

    def __init__(self, model_type: str, config_manager: Any):
        self.model_type = model_type.lower()
        self.config_manager = config_manager
        self.search_space = self.config_manager.config.get("search_spaces", {}).get("traditional", {}).get(self.model_type, {})
        
    def _get_model_class(self):
        if self.model_type == "svr":
            from sklearn.svm import SVR
            return SVR
        elif self.model_type == "xgb":
            from xgboost import XGBRegressor
            return XGBRegressor
        elif self.model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor
        elif self.model_type == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor
        elif self.model_type == "gpr":
            from sklearn.gaussian_process import GaussianProcessRegressor
            return GaussianProcessRegressor
        else:
            raise ValueError(f"Unsupported model type for optimization: {self.model_type}")

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters from the configured search space, honoring conditions."""
        params = {}
        for param_name, config in self.search_space.items():
            # Skip if conditions are not satisfied
            condition = config.get("condition")
            if condition:
                satisfied = True
                for cond_param, cond_val in condition.items():
                    current = params.get(cond_param)
                    if isinstance(cond_val, list):
                        if current not in cond_val:
                            satisfied = False
                    else:
                        if current != cond_val:
                            satisfied = False
                if not satisfied:
                    continue

            param_type = config.get("type")
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"],
                    step=config.get("step", 1),
                    log=config.get("log", False),
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"],
                    step=config.get("step"),
                    log=config.get("log", False),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    config["choices"],
                )

        # Fixed params that are not in search space but required
        if self.model_type == "xgb":
            params.update({"objective": "reg:squarederror", "n_jobs": 1, "random_state": 42})
        elif self.model_type == "mlp":
            params.update({"max_iter": 1000, "random_state": 42})
        elif self.model_type == "rf":
            params.update({"random_state": 42})

        return params

    def optimize(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100, n_jobs: int = 1, study_name: str = None, storage: str = None) -> Dict[str, Any]:
        """
        Run the optimization.
        """
        model_class = self._get_model_class()
        cv_folds = self.config_manager.get_training_params().get("cv_folds", 5)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        def objective(trial):
            params = self._suggest_params(trial)
            model = model_class(**params)
            
            # Handle n_jobs for cross_val_score
            # Some models like XGBoost handle parallelism internally, so we might want n_jobs=1 for CV
            # Others like SVR are single threaded (mostly), so we want n_jobs=-1 for CV
            cv_n_jobs = n_jobs
            if self.model_type == "xgb":
                 # XGBoost can use threads. If we run parallel CV, we should limit XGB threads.
                 # Here we set XGB n_jobs=1 in _suggest_params and use parallel CV.
                 pass
            elif self.model_type == "svr":
                 # SVR is single threaded
                 pass
            
            scores = cross_val_score(
                model, X, y, 
                cv=kf, 
                scoring='neg_mean_absolute_error', 
                n_jobs=cv_n_jobs
            )
            return scores.mean()

        if study_name is None:
            study_name = f"{self.model_type}_optimization"
            
        study = optuna.create_study(
            direction='maximize', 
            study_name=study_name, 
            storage=storage, 
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best params: {study.best_params}")
        print(f"Best value: {study.best_value}")
        
        return study.best_params

if __name__ == "__main__":
    # Entry point for remote execution
    import argparse
    from .config import MLConfigManager
    
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimization")
    parser.add_argument("--model", type=str, required=True, help="Model type (xgb, svr, etc.)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data pickle file (X, y)")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="Optuna storage URL")
    
    args = parser.parse_args()
    
    # Load data
    data = joblib.load(args.data_path)
    if isinstance(data, tuple):
        X, y = data
    else:
        # Assume dict or similar structure if needed, but tuple (X, y) is standard
        X = data['X']
        y = data['y']
        
    config = MLConfigManager()
    optimizer = HyperparameterOptimizer(args.model, config)
    
    best_params = optimizer.optimize(
        X, y, 
        n_trials=args.n_trials, 
        n_jobs=args.n_jobs,
        study_name=f"{args.model}_opt",
        storage=args.storage
    )
    
    # Train final model and save
    print("Training final model with best params...")
    model_class = optimizer._get_model_class()
    
    # Re-construct params (need to handle fixed params again or merge)
    # Ideally _suggest_params logic should be reusable or we just trust best_params + defaults
    # For simplicity, we re-instantiate with best_params. 
    # Note: best_params from optuna only contains suggested params. 
    # We need to add back the fixed params (like random_state, n_jobs, objective)
    
    final_params = best_params.copy()
    if args.model == "xgb":
        final_params.update({"objective": "reg:squarederror", "n_jobs": 12, "random_state": 42})
    elif args.model == "mlp":
        final_params.update({"max_iter": 1000, "random_state": 42})
    elif args.model == "rf":
        final_params.update({"random_state": 42})
        
    model = model_class(**final_params)
    model.fit(X, y)
    
    joblib.dump(model, f"{args.model}_best_model.pkl")
    print(f"Model saved to {args.model}_best_model.pkl")
