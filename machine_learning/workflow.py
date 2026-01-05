import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from sklearn.model_selection import train_test_split
import joblib
import json

from .config import MLConfigManager
from .data_loader import MLTrainDataBuilder
from .analysis.metrics import MetricsCalculator
from .job_generator import JobGenerator

class MLWorkflow:
    """
    Main workflow for Machine Learning tasks.
    """
    
    TRADITIONAL_MODELS = ["svr", "gpr", "xgb", "rf", "mlp"]
    GNN_MODELS = ["schnet", "dimenet_pp"]

    def _get_model_class(self, model_type: str):
        """Lazy load model classes to avoid heavy imports at startup."""
        if model_type == "svr":
            from .models.traditional.svr import SVRModel
            return SVRModel
        elif model_type == "gpr":
            from .models.traditional.gpr import GPRModel
            return GPRModel
        elif model_type == "xgb":
            from .models.traditional.xgb import XGBModel
            return XGBModel
        elif model_type == "rf":
            from .models.traditional.rf import RFModel
            return RFModel
        elif model_type == "mlp":
            from .models.traditional.mlp import MLPModel
            return MLPModel
        elif model_type == "schnet":
            from .models.gnn.schnet import SchNetModel
            return SchNetModel
        elif model_type == "dimenet_pp":
            from .models.gnn.dimenet_pp import DimeNetPPModel
            return DimeNetPPModel
        else:
            return None

    def __init__(self, config_path: Optional[str] = None, elements_csv_path: str = "elements_properties_all.csv"):
        self.config_manager = MLConfigManager(config_path)
        
        # Load elements data
        if os.path.exists(elements_csv_path):
            self.elements_df = pd.read_csv(elements_csv_path)
        else:
            # Try to find it in the workspace if relative path fails
            # Assuming standard location based on workspace info
            possible_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "discriptor", "elements_properties_all.csv")
            if os.path.exists(possible_path):
                self.elements_df = pd.read_csv(possible_path)
            else:
                # Fallback for when running in a different context where file might not be needed immediately
                print(f"Warning: Elements CSV not found at {elements_csv_path} or {possible_path}")
                self.elements_df = None
                
        if self.elements_df is not None:
            self.data_builder = MLTrainDataBuilder(self.elements_df)
        else:
            self.data_builder = None
            
        self.model = None
        self.model_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def prepare_data(
        self,
        structures: List[Dict],
        targets: Dict[str, float],
        model_name: str,
        atom_indices: List[int] = [0], # Default to first atom if not specified
        parse_structure_func: Callable = None
    ):
        """
        Prepare data for training.
        """
        if self.data_builder is None:
            raise ValueError("Data builder not initialized. Check elements CSV path.")

        self.model_type = model_name.lower()
        
        if self.model_type in self.TRADITIONAL_MODELS:
            print("Building tabular dataset for traditional ML...")
            X, y, filenames = self.data_builder.build_tabular_dataset(
                structures, targets, atom_indices, parse_structure_func=parse_structure_func
            )
            self.feature_names = X.columns.tolist()
            
            # Split data
            train_params = self.config_manager.get_training_params()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, 
                test_size=train_params.get("test_size", 0.2),
                random_state=train_params.get("random_state", 42)
            )
            
        elif self.model_type in self.GNN_MODELS:
            print("Building graph dataset for GNN...")
            dataset = self.data_builder.build_graph_dataset(
                structures, targets, parse_structure_func=parse_structure_func
            )
            
            # Split data
            train_params = self.config_manager.get_training_params()
            train_data, test_data = train_test_split(
                dataset,
                test_size=train_params.get("test_size", 0.2),
                random_state=train_params.get("random_state", 42)
            )
            self.X_train = train_data # List of dicts
            self.X_test = test_data   # List of dicts
            # y is embedded in X for GNNs in this implementation
            
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def optimize_hyperparameters(self, n_trials: int = 100, n_jobs: int = 1):
        """
        Run hyperparameter optimization locally.
        """
        if self.model_type not in self.TRADITIONAL_MODELS:
            print("Optimization currently only supported for traditional models.")
            return

        from .optimizer import HyperparameterOptimizer
        print(f"Starting optimization for {self.model_type} with {n_trials} trials...")
        optimizer = HyperparameterOptimizer(self.model_type, self.config_manager)
        
        best_params = optimizer.optimize(
            self.X_train, self.y_train,
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        
        # Update config with best params
        # Note: This updates the in-memory config for the current session
        if "models" not in self.config_manager.config:
            self.config_manager.config["models"] = {}
        if "traditional" not in self.config_manager.config["models"]:
            self.config_manager.config["models"]["traditional"] = {}
            
        # Merge best params with existing defaults to keep non-optimized params
        current_params = self.config_manager.config["models"]["traditional"].get(self.model_type, {})
        current_params.update(best_params)
        self.config_manager.config["models"]["traditional"][self.model_type] = current_params
        
        print("Optimization complete. Best parameters loaded into configuration.")

    def _resolve_output_dir(self, output_dir: Optional[str]) -> str:
        if output_dir:
            return output_dir
        root = self.config_manager.config.get("local_paths", {}).get("results_root") or "./outputs"
        return os.path.join(root, "machine_learning")

    def prepare_job_files(self, output_dir: str, job_name: str = "ml_job", **slurm_kwargs):
        """
        Generate files for remote job submission.
        """
        output_dir = self._resolve_output_dir(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Save Data
        data_path = os.path.join(output_dir, "train_data.pkl")
        joblib.dump((self.X_train, self.y_train, self.X_test, self.y_test), data_path)
        
        # 2. Get Params
        params = self.config_manager.get_model_params(
            "traditional" if self.model_type in self.TRADITIONAL_MODELS else "gnn",
            self.model_type
        )
        
        # 3. Generate Files
        generator = JobGenerator(output_dir)
        files = generator.generate_job_files(
            job_name=job_name,
            model_name=self.model_type,
            params=params,
            **slurm_kwargs
        )
        
        generator.save_job_files(files)
        return files

    def submit_optimization_job(self, output_dir: str = "remote_jobs", n_trials: int = 100, n_jobs: int = 1):
        """
        Generate files for remote optimization job.
        """
        if self.model_type not in self.TRADITIONAL_MODELS:
            print("Remote optimization currently only supported for traditional models.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Save data
        data_file = "train_data.pkl"
        data_path = os.path.join(output_dir, data_file)
        joblib.dump((self.X_train, self.y_train), data_path)
        print(f"Data saved to {data_path}")
        
        # 2. Generate SLURM script
        generator = JobGenerator(output_dir)
        slurm_path = generator.generate_optimization_job(
            model_name=self.model_type,
            data_file=data_file,
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        
        print(f"Job files generated in {output_dir}")
        print(f"To submit: sbatch {slurm_path}")
        print("Note: Ensure 'machine_learning' module is available in PYTHONPATH on the remote server.")

    def train(self):
        """
        Train the model.
        """
        if self.model_type is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
            
        model_class = self._get_model_class(self.model_type)
        if not model_class:
            raise ValueError(f"Model class not found for {self.model_type}")
            
        # Get model params
        params = self.config_manager.get_model_params(
            "traditional" if self.model_type in self.TRADITIONAL_MODELS else "gnn",
            self.model_type
        )
        
        print(f"Initializing {self.model_type} with params: {params}")
        self.model = model_class(params)
        
        print("Training model...")
        if self.model_type in self.TRADITIONAL_MODELS:
            self.model.train(self.X_train, self.y_train)
        else:
            self.model.train(self.X_train, val_dataset=self.X_test)
            
        print("Training complete.")

    def evaluate(self, output_dir: str = "results"):
        """
        Evaluate the model and save results.
        """
        output_dir = self._resolve_output_dir(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print("Evaluating model...")
        
        if self.model_type in self.TRADITIONAL_MODELS:
            y_pred_train = self.model.predict(self.X_train)
            y_pred_test = self.model.predict(self.X_test)
            
            train_metrics = MetricsCalculator.calculate_metrics(self.y_train, y_pred_train)
            test_metrics = MetricsCalculator.calculate_metrics(self.y_test, y_pred_test)
            
            # Save metrics
            metrics = {
                "train": train_metrics,
                "test": test_metrics
            }
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                
            print(f"Train Metrics: {train_metrics}")
            print(f"Test Metrics: {test_metrics}")
            # SHAP Analysis
            print("Running SHAP analysis...")
            try:
                from .analysis.shap_analysis import ShapAnalyzer
                shap_analyzer = ShapAnalyzer(self.model.model, self.X_train)
                shap_values = shap_analyzer.calculate_shap_values(self.X_test)
                if shap_values is not None:
                    shap_analyzer.plot_summary(
                        shap_values, 
                        self.X_test, 
                        save_path=os.path.join(output_dir, "shap_summary.png")
                    )
                    # Save feature importance if possible
                    if hasattr(self.model.model, "feature_importances_"):
                        fi = pd.DataFrame({
                            "feature": self.feature_names,
                            "importance": self.model.model.feature_importances_
                        }).sort_values("importance", ascending=False)
                        fi.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
                
        else: # GNN
            # For GNN, we need to extract targets from the dataset lists
            y_train = np.array([d['target'] for d in self.X_train])
            y_test = np.array([d['target'] for d in self.X_test])
            
            y_pred_train = self.model.predict(self.X_train)
            y_pred_test = self.model.predict(self.X_test)
            
            train_metrics = MetricsCalculator.calculate_metrics(y_train, y_pred_train)
            test_metrics = MetricsCalculator.calculate_metrics(y_test, y_pred_test)
            
            metrics = {
                "train": train_metrics,
                "test": test_metrics
            }
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                
            print(f"Train Metrics: {train_metrics}")
            print(f"Test Metrics: {test_metrics}")
            
        # Save predictions
        if self.model_type in self.TRADITIONAL_MODELS:
            results_df = pd.DataFrame({
                "y_true": self.y_test,
                "y_pred": y_pred_test
            })
            results_df.to_csv(os.path.join(output_dir, "predictions_test.csv"), index=False)
        else:
             results_df = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred_test
            })
             results_df.to_csv(os.path.join(output_dir, "predictions_test.csv"), index=False)

    def save_model(self, path: str):
        """Save the trained model."""
        if self.model_type in self.TRADITIONAL_MODELS:
            joblib.dump(self.model, path)
        else:
            torch.save(self.model.model.state_dict(), path)
