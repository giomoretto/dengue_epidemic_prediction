"""One-step-ahead predictions over the full history. The weather of the predicted week is
unknown at forecast time, so it is drawn from the climatology instead of using the observation."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_definition.weekly_model import get_climate_sample, load_model
from utils.constants import CLIMATE_VARS
from utils.epi_calendar import ONE_WEEK, epi_week_from_date
from utils.feature_extraction import CURRENT_LEVEL, NEXT_LEVEL, lagged_name
from utils.types import City

OBSERVED_COLOR = "#52514e"
PREDICTED_COLOR = "#2a78d6"
GRID_COLOR = "#e1e0d9"


def main(
    df_features: pd.DataFrame,
    climate_dict: dict,
    model_definition: str,
    seed: int = 0,
):
    """Plot one-step-ahead predictions against the observations for all cities.

    Args:
        df_features: Weekly feature table (one row per city and week).
        climate_dict: Weekly climatology with keys "mean" and "cov".
        model_definition: Registered MLflow model (e.g. "next_week_prediction/latest").
        seed: Seed of the random number generator.
    """
    model, feature_columns, _, _ = load_model(model_definition)
    rng = np.random.default_rng(seed)
    next_week_climate = [lagged_name(var, 1) for var in CLIMATE_VARS]
    observed_features = [col for col in feature_columns if col not in next_week_climate]

    fig, axes = plt.subplots(len(City), 1, figsize=(12, 16), sharex=True)

    for ax, city in zip(axes, City):
        city_df = df_features[df_features["city_name"] == city.value].sort_values("date")
        target_dates = city_df["date"] + ONE_WEEK

        # Replace the observed weather of week t+1 by a sample from its climatology
        features = city_df.copy()
        _, target_epi_weeks = epi_week_from_date(target_dates)
        for epi_week in np.unique(target_epi_weeks):
            rows = target_epi_weeks == epi_week
            features.loc[rows, next_week_climate] = get_climate_sample(
                city, epi_week, climate_dict["mean"], climate_dict["cov"], int(rows.sum()), rng
            )

        # The model predicts the weekly change: x_(t+1) = x_(t) + f(...)
        complete = city_df[observed_features].notna().all(axis=1).to_numpy()
        predicted = np.full(len(city_df), np.nan)
        predicted[complete] = city_df.loc[complete, CURRENT_LEVEL] + model.predict(
            features.loc[complete, feature_columns]
        )

        ax.plot(target_dates, city_df[NEXT_LEVEL], label="Observed",
                color=OBSERVED_COLOR, linewidth=0.8)
        ax.plot(target_dates, predicted, label="Predicted (sampled weather)",
                color=PREDICTED_COLOR, linewidth=0.8)
        ax.set_ylabel("log_p_inc100k")
        ax.set_title(city.value)
        ax.legend(fontsize=8)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year")
    fig.suptitle("Observed vs One-step-ahead Prediction", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    from utils.run_complete_pipeline import (
        FEATURE_SETTINGS,
        FILENAME,
        TRAIN_SETTINGS,
        run_complete_pipeline,
    )

    MODEL_DEFINITION = "next_week_prediction/latest"

    df_features, climate_dict = run_complete_pipeline(
        filename=FILENAME,
        feature_settings=FEATURE_SETTINGS,
        train_settings={**TRAIN_SETTINGS, "train_model": False},
    )

    main(
        df_features=df_features,
        climate_dict=climate_dict,
        model_definition=MODEL_DEFINITION,
    )
