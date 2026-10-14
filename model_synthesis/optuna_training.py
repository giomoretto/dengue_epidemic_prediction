"""Hyperparameter search with Optuna on backtesting folds, final fit, hold-out test and MLflow logging,
shared by the Random Forest and XGBoost models."""

from collections.abc import Callable

import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from model_definition.error_model import TURN_OF_YEAR_WEEKS, ErrorModel, residual_pairs
from model_synthesis.backtesting import BacktestSplits, backtest, persistence_skill, rows_for_years
from utils.config import MLFLOW_TRACKING_URI
from utils.feature_extraction import CURRENT_LEVEL, TARGET_COLUMN, build_feature_columns, weekly_cases
from utils.plots import (
    plot_error_model,
    plot_residuals,
    plot_predicted_vs_true,
    plot_feature_importance,
    plot_time_series_predictions,
)


def train_with_optuna(
    *,
    model_label: str,
    experiment_name: str,
    make_model: Callable[[dict], object],
    suggest_params: Callable[[optuna.Trial], dict],
    log_model: Callable,
    model_artifact_name: str,
    df_features: pd.DataFrame,
    dataset_name: str,
    feature_columns: list[str],
    n_lags: int,
    splits: BacktestSplits,
    n_trials: int,
    random_state: int,
    registered_model_name: str | None = None,
):
    """Select hyperparameters by backtesting, refit on all development years and test on the hold-out years.

    Every Optuna trial is scored by the persistence skill of its out-of-fold predictions over all
    backtesting folds. The best parameters are refitted on all years before the test years and
    evaluated once on the hold-out test years. The out-of-fold residuals of the best parameters
    are used to fit the error model of the Monte-Carlo simulation.

    Args:
        model_label: Model name used in plot titles (e.g. "Random Forest").
        experiment_name: MLflow experiment name.
        make_model: Returns an unfitted model for a dict of hyperparameters.
        suggest_params: Draws the hyperparameters of an Optuna trial.
        log_model: MLflow flavor's log_model function (e.g. mlflow.sklearn.log_model).
        model_artifact_name: Name of the logged model in the MLflow run.
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
    # Build suffixed feature columns from base names
    feature_columns = build_feature_columns(feature_columns, n_lags)

    # Drop weeks without complete history or without a next week (series ends, data gaps)
    df = df_features.dropna(subset=[*feature_columns, TARGET_COLUMN])
    development = rows_for_years(df, splits.development_years)
    test = rows_for_years(df, splits.test_years)

    for i, fold in enumerate(splits.folds):
        print(f"Fold {i}: train {fold.train_years[0]}-{fold.train_years[-1]}, "
              f"validate {fold.validation_years[0]}-{fold.validation_years[-1]}")
    print(f"Test: {splits.test_years[0]}-{splits.test_years[-1]} ({len(test)} rows)")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    def objective(trial):
        """Persistence skill of the out-of-fold predictions (Optuna maximizes it)."""
        params = suggest_params(trial)
        result = backtest(lambda: make_model(params), df, feature_columns, splits.folds)
        trial.set_user_attr("fold_skills", result["fold_skills"])
        return result["skill"]

    # Run Optuna optimization
    print("Starting Optuna hyperparameter search...")
    study = optuna.create_study(
        direction="maximize",
        study_name=experiment_name,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptimization complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best backtest skill: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    # Train final model with best parameters and log to MLflow
    with mlflow.start_run():
        best_params = study.best_params
        best_backtest = backtest(lambda: make_model(best_params), df, feature_columns, splits.folds)

        # Error model from the out-of-fold residuals: honest one-step errors on unseen years
        residuals = best_backtest["residuals"]
        residual_rows = df.loc[residuals.index]
        residual_table = pd.DataFrame({
            "city_name": residual_rows["city_name"],
            "date": residual_rows["date"],
            "epi_week": residual_rows["epi_week"],
            "cases": weekly_cases(residual_rows[CURRENT_LEVEL], residual_rows["pop"]),
            "residual": residuals,
        })
        error_model = ErrorModel.fit(residual_table)

        model = make_model(best_params)
        model.fit(development[feature_columns], development[TARGET_COLUMN])

        delta_true = test[TARGET_COLUMN].to_numpy()
        delta_pred = model.predict(test[feature_columns])

        metrics = {
            "test_skill": persistence_skill(delta_true, delta_pred),
            "test_rmse": root_mean_squared_error(delta_true, delta_pred),
            "test_mae": mean_absolute_error(delta_true, delta_pred),
            "test_rmse_persistence": float(np.sqrt(np.mean(delta_true**2))),
            "backtest_skill": best_backtest["skill"],
            "backtest_rmse": float(np.sqrt(np.mean(residuals**2))),
            **{f"backtest_skill_fold_{i}": skill for i, skill in enumerate(best_backtest["fold_skills"])},
            "error_sigma_eta": error_model.sigma_eta,
            "error_phi": error_model.phi,
            "error_tau2": error_model.tau2,
        }

        # Log the dataset, parameters and metrics
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_input(
            mlflow.data.from_pandas(development, name=dataset_name, targets=TARGET_COLUMN), context="training"
        )
        mlflow.log_input(mlflow.data.from_pandas(test, name=dataset_name, targets=TARGET_COLUMN), context="testing")
        mlflow.log_param("model_type", model_label)
        mlflow.log_params(best_params)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features", feature_columns)
        mlflow.log_param("n_lags", n_lags)
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("folds", [(f.train_years[-1], f.validation_years) for f in splits.folds])
        mlflow.log_param("development_years", str(splits.development_years))
        mlflow.log_param("test_years", str(splits.test_years))
        mlflow.log_param("city_encoding", "CITY_ENCODING")
        mlflow.log_param("optimization_trials", len(study.trials))
        mlflow.log_metrics(metrics)

        # Log the model and its error model (loaded together by model_definition.weekly_model.load_model)
        signature = infer_signature(development[feature_columns], model.predict(development[feature_columns]))
        log_model(
            model,
            name=model_artifact_name,
            signature=signature,
            registered_model_name=registered_model_name,
        )
        mlflow.log_dict(error_model.to_dict(), "error_model.json")

        print("\nFinal Model Performance:")
        print(f"Features: {feature_columns}")
        print(f"Best Parameters: {best_params}")
        print(f"Metrics: {metrics}")

        # Save plots (weekly change of the log incidence on the test years)
        fig_residuals = plot_residuals(delta_true, delta_pred)
        mlflow.log_figure(fig_residuals, "residual_plot.png")

        fig_pred_true = plot_predicted_vs_true(delta_true, delta_pred)
        mlflow.log_figure(fig_pred_true, "predicted_vs_true.png")

        fig_feat_imp = plot_feature_importance(
            model.feature_importances_,
            feature_columns,
            f"{model_label} Feature Importance",
        )
        mlflow.log_figure(fig_feat_imp, "feature_importance.png")

        fig_ts = plot_time_series_predictions(df, model, feature_columns, splits.test_years)
        mlflow.log_figure(fig_ts, "time_series_predictions.png")

        fig_error = plot_error_model(
            residual_table,
            residual_table["epi_week"].isin(TURN_OF_YEAR_WEEKS),
            residual_pairs(residual_table),
            error_model,
        )
        mlflow.log_figure(fig_error, "error_model.png")

        print("\nPlots saved to MLflow")

        # Save Optuna study visualization
        try:
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_optimization = optuna.visualization.plot_optimization_history(study)

            mlflow.log_figure(fig_importance, "optuna_param_importances.html")
            mlflow.log_figure(fig_optimization, "optuna_optimization_history.html")
            print("Optuna visualizations saved to MLflow")
        except Exception as e:
            print(f"\nCould not save Optuna visualizations: {e}")

    return model, study, metrics
