import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import optuna

# Allow overriding trial count via env
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "2"))


def load_config(default_model: str):
    """Load optional config.json to pass hyperparameters."""
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            cfg = json.load(f)
        return cfg.get("model_name", default_model), cfg.get("params", {})
    return default_model, {}


def make_model(name: str, params: dict):
    registry = {
        "xgb": lambda p: xgb.XGBRegressor(**p),
        "rf": lambda p: RandomForestRegressor(**p),
        "svr": lambda p: SVR(**p),
        "gpr": lambda p: GaussianProcessRegressor(**p),
        "krr": lambda p: KernelRidge(**p),
        "mlp": lambda p: MLPRegressor(**p),
    }
    if name not in registry:
        raise ValueError(f"Unsupported model: {name}")
    return registry[name](params)


def xgboost_safe(params: dict):
    """Tiny helper to keep XGB import in one place if we later swap backend."""
    return xgb.XGBRegressor(**params)


def run_optuna_mlp(X, y, n_trials=N_TRIALS):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(64, 32), (128, 64), (256, 128), (128, 64, 32)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": trial.suggest_categorical("solver", ["adam"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
            "max_iter": 1000,
            "random_state": 42,
        }
        model = MLPRegressor(**params)
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", study_name="MLP_opt", storage="sqlite:///optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value


def run_optuna_svr(X, y, n_trials=N_TRIALS):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params = {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1e0, log=True),
            "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
            "kernel": kernel,
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
            params["coef0"] = trial.suggest_float("coef0", 0.0, 1.0)
        if kernel in ("rbf", "poly"):
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0])

        model = SVR(**params)
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", study_name="SVR_opt", storage="sqlite:///optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value


def run_optuna_xgb(X, y, n_trials=N_TRIALS):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_categorical("subsample", [0.5, 0.6, 0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.5, 0.6, 0.8, 1.0]),
            "colsample_bylevel": trial.suggest_categorical("colsample_bylevel", [0.5, 0.6, 0.8, 1.0]),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
        }
        model = xgboost_safe(params)
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", study_name="XGB_opt", storage="sqlite:///optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = study.best_params
    best_params.update({"objective": "reg:squarederror", "random_state": 42, "n_jobs": 1})
    return best_params, study.best_value


def run_optuna_krr(X, y, n_trials=N_TRIALS):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
        params = {"kernel": kernel, "alpha": alpha}
        if kernel in {"rbf", "poly"}:
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 3)
            params["coef0"] = trial.suggest_float("coef0", 0.0, 5.0)

        model = KernelRidge(**params)
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1, error_score=np.nan)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", study_name="KRR_opt", storage="sqlite:///optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value


def run_optuna_gpr(X, y, n_trials=N_TRIALS):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kernel_dict = {
        "RBF_0.1": RBF(length_scale=0.1),
        "RBF_1": RBF(length_scale=1.0),
        "RBF_10": RBF(length_scale=10.0),
        "Matern_0.5": Matern(length_scale=1.0, nu=0.5),
        "Matern_1.5": Matern(length_scale=1.0, nu=1.5),
        "RQ": RationalQuadratic(length_scale=1.0, alpha=1.0),
        "RBF+White": RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
    }

    def objective(trial):
        kernel_tag = trial.suggest_categorical("kernel", list(kernel_dict.keys()))
        alpha = trial.suggest_float("alpha", 1e-12, 1e-4, log=True)
        n_restarts = trial.suggest_int("n_restarts_optimizer", 0, 5)
        kernel = kernel_dict[kernel_tag]

        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            optimizer="fmin_l_bfgs_b",
            random_state=42,
        )
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1, error_score=np.nan)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", study_name="GPR_opt", storage="sqlite:///optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = study.best_params
    # Convert kernel tag back to actual kernel and add deterministic defaults
    best_params["kernel"] = kernel_dict[best_params["kernel"]]
    best_params.setdefault("optimizer", "fmin_l_bfgs_b")
    best_params.setdefault("random_state", 42)
    return best_params, study.best_value


def run_optuna_rf(X, y, n_trials=N_TRIALS):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=10),
            "max_depth": trial.suggest_int("max_depth", 3, 20, step=1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestRegressor(**params)
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", study_name="RF_opt", storage="sqlite:///optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1})
    return best_params, study.best_value


def main():
    data_path = "train_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found")

    use_graph = False
    if os.path.exists("train_data.pkl"):
        X_train, y_train, X_test, y_test = joblib.load("train_data.pkl")
        if isinstance(X_train, list) and X_train and isinstance(X_train[0], dict) and "atomic_numbers" in X_train[0]:
            use_graph = True
    else:
        df = pd.read_csv(data_path)
        target = df.columns[-1]
        if "filename" in df.columns:
            X = df.drop(columns=["filename", target])
        else:
            X = df.iloc[:, :-1]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_name, params = load_config("{model_name}")

    if use_graph:
        from machine_learning.models.gnn.schnet import SchNetModel
        from machine_learning.models.gnn.dimenet_pp import DimeNetPPModel

        gnn_registry = {
            "schnet": SchNetModel,
            "dimenet_pp": DimeNetPPModel,
        }
        if model_name not in gnn_registry:
            raise ValueError(f"Unsupported GNN model: {model_name}")

        ModelCls = gnn_registry[model_name]
        model = ModelCls(params)
        model.train(X_train, val_dataset=X_test)
        preds = model.predict(X_test)
        extra_info = {}
    else:
        optuna_registry = {
            "mlp": run_optuna_mlp,
            "svr": run_optuna_svr,
            "xgb": run_optuna_xgb,
            "krr": run_optuna_krr,
            "gpr": run_optuna_gpr,
            "rf": run_optuna_rf,
        }

        if model_name in optuna_registry:
            merged_X = pd.concat([X_train, X_test], axis=0)
            merged_y = pd.concat([y_train, y_test], axis=0)
            best_params, best_score = optuna_registry[model_name](merged_X, merged_y, n_trials=N_TRIALS)
            params.update(best_params)
            if model_name == "mlp":
                params.setdefault("max_iter", 1000)
                params.setdefault("random_state", 42)
            if model_name in {"xgb", "rf"}:
                params.setdefault("random_state", 42)
            extra_info = {"optuna_best_cv_mae": float(best_score)}
        else:
            extra_info = {}

        model = make_model(model_name, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        **extra_info,
    }

    out = pd.DataFrame({"y_true": y_test, "y_pred": preds})
    # root = os.getenv("RESULTS_ROOT", "./outputs")
    # out_dir = Path(root) / "machine_learning"
    # out_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv("results.csv", index=False)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Metrics: {metrics}")


if __name__ == "__main__":
    main()
