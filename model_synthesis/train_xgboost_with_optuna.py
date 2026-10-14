import mlflow
import optuna
import pandas as pd
from xgboost import XGBRegressor

from model_synthesis.backtesting import BacktestSplits
from model_synthesis.optuna_training import train_with_optuna


def _suggest_params(trial: optuna.Trial) -> dict:
    """Hyperparameter search space of XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }


def train_xgboost_with_optuna(
    df_features: pd.DataFrame,
    dataset_name: str,
    feature_columns: list[str],
    n_lags: int,
    splits: BacktestSplits,
    n_trials: int,
    random_state: int,
    registered_model_name: str | None = None,
):
    """Train an XGBoost model for the weekly change of the log incidence, with Optuna hyperparameter
    optimization on backtesting folds.

    Args:
        df_features: Weekly feature table (see utils.feature_extraction.run_feature_pipeline).
        dataset_name: Name of the raw dataset the features were built from, logged with the run.
        feature_columns: Base feature names (converted to suffixed columns via build_feature_columns).
        n_lags: Number of lagged weeks used as features in addition to week t.
        splits: Backtesting folds, development years and test years.
        n_trials: Number of Optuna optimization trials.
        random_state: Random seed for reproducibility.
        registered_model_name: If given, register the final model under this name in MLflow.

    Returns:
        tuple: (trained_model, study, metrics_dict)
    """
    return train_with_optuna(
        model_label="XGBoost",
        experiment_name="dengue_xgboost_optuna",
        make_model=lambda params: XGBRegressor(**params, random_state=random_state, n_jobs=-1),
        suggest_params=_suggest_params,
        log_model=mlflow.xgboost.log_model,
        model_artifact_name="xgboost_model_optimized",
        df_features=df_features,
        dataset_name=dataset_name,
        feature_columns=feature_columns,
        n_lags=n_lags,
        splits=splits,
        n_trials=n_trials,
        random_state=random_state,
        registered_model_name=registered_model_name,
    )


if __name__ == "__main__":
    from model_synthesis.backtesting import make_backtest_splits
    from utils.config import PROJECT_ROOT
    from utils.run_complete_pipeline import FEATURE_SETTINGS, FILENAME, TRAIN_SETTINGS

    # Feature table written by data_preparation/write_weekly_features.py
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features_weekly.parquet")
    splits = make_backtest_splits(df["year"].unique(), **TRAIN_SETTINGS["backtesting"])

    train_xgboost_with_optuna(
        df_features=df,
        dataset_name=FILENAME,
        feature_columns=TRAIN_SETTINGS["feature_columns"],
        n_lags=FEATURE_SETTINGS["n_lags"],
        splits=splits,
        n_trials=50,
        random_state=TRAIN_SETTINGS["random_state"],
    )
