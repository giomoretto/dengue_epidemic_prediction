"Run a complete pipeline: loading, weekly feature extraction, backtested model fit, and climate tables"

from data_preparation import csv_to_parquet
from model_synthesis import TRAINERS
from model_synthesis.backtesting import make_backtest_splits
from utils.climate_tables import create_climate_tables
from utils.constants import CLIMATE_VARS
from utils.feature_extraction import run_feature_pipeline

# Settings shared by training, one-step validation and Monte-Carlo simulation
FILENAME = "infodengue_capitals_subsetBR"

FEATURE_SETTINGS = {
    # Lagged weeks t-1, ..., t-n_lags used as features in addition to week t (0: only week t)
    "n_lags": 0,
    "columns": CLIMATE_VARS + ["p_inc100k", "pop"],
}

TRAIN_SETTINGS = {
    "train_model": True,
    "model": "random_forest",  # "random_forest" or "xgboost"
    "optuna_trials": 150,
    "random_state": 42,
    "feature_columns": CLIMATE_VARS + ["city_encoded", "epi_week", "log_p_inc100k"],
    "registered_model_name": "next_week_prediction",
    "backtesting": {
        "n_test_years": 2,  # hold-out: the last two years, used only for the final test
        "n_validation_years": 2,  # years ahead validated in each backtesting fold
        "n_folds": 4,  # expanding-window folds before the hold-out years
    },
}


def run_complete_pipeline(
    filename: str,
    feature_settings: dict,
    train_settings: dict,
):
    """Run the complete pipeline on weekly data: convert CSV to Parquet, build features, train with backtesting, and compute climate tables.

    Args:
        filename (str): Name of the dataset file (without extension).
        feature_settings (dict): Columns to keep ("columns") and number of lagged weeks ("n_lags").
        train_settings (dict): Settings for training the model, including whether to train, hyperparameter search and backtesting.

    Returns:
        tuple: (df_features, climate_dict) - weekly feature table and weekly climatology of the development years
    """
    # Step 1: Convert CSV to Parquet
    df_raw = csv_to_parquet(filename)
    print(f"Raw data loaded from {filename}.csv and converted to Parquet format.")

    # Step 2: Weekly panel with lagged features and target
    df_features = run_feature_pipeline(
        df=df_raw,
        columns=feature_settings["columns"],
        n_lags=feature_settings["n_lags"],
    )
    print(
        f"Weekly features created with n_lags={feature_settings['n_lags']} for columns: {feature_settings['columns']}."
    )

    # Step 3: Backtesting folds and hold-out test years
    splits = make_backtest_splits(df_features["year"].unique(), **train_settings["backtesting"])

    # Step 4: Optional - Train model with prepared data
    if train_settings["train_model"]:
        if train_settings["model"] not in TRAINERS:
            raise ValueError(f"Unknown model {train_settings['model']!r}, choose one of {list(TRAINERS)}")

        model, study, metrics = TRAINERS[train_settings["model"]](
            df_features=df_features,
            dataset_name=filename,
            feature_columns=train_settings["feature_columns"],
            n_lags=feature_settings["n_lags"],
            splits=splits,
            n_trials=train_settings["optuna_trials"],
            random_state=train_settings["random_state"],
            registered_model_name=train_settings["registered_model_name"],
        )
        print(
            f"\nModel training complete! Skill vs persistence - backtest: {metrics['backtest_skill']:.3f}, "
            f"test: {metrics['test_skill']:.3f}"
        )

    # Step 5: Create climate tables from the development years (all years before the test years)
    climate_dict = create_climate_tables(
        df_weekly=df_features[df_features["year"].isin(splits.development_years)], columns=CLIMATE_VARS
    )
    print("Climate tables created from the development years.")

    return df_features, climate_dict


if __name__ == "__main__":
    df_features, climate_dict = run_complete_pipeline(
        filename=FILENAME,
        feature_settings=FEATURE_SETTINGS,
        train_settings=TRAIN_SETTINGS,
    )
