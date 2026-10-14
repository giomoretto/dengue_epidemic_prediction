"""Calibration of the Monte-Carlo forecast: how often do the observations fall inside the predicted intervals?

Forecasts are started every few weeks of the given years in all cities and compared with the
observed log incidence of the following weeks. For well calibrated forecasts, a central x %
interval contains about x % of the observations at every horizon.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model_definition.weekly_model import get_history, load_model, predict_trajectories
from utils.config import PROJECT_ROOT
from utils.types import City

INTERVALS = (0.5, 0.75, 0.95)  # central prediction intervals
OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction"


def main(
    df_features: pd.DataFrame,
    climate_mean: pd.DataFrame,
    climate_cov: pd.DataFrame,
    start_years: list[int],
    model_definition: str,
    horizon: int = 26,
    start_every: int = 3,
    n_samples: int = 300,
    include_model_error: bool = True,
    output_dir: Path = OUTPUT_DIR,
    seed: int = 1,
) -> pd.DataFrame:
    """Share of observations inside the central prediction intervals, per forecast horizon.

    Args:
        df_features: Weekly feature table (one row per city and week).
        climate_mean: Mean climate per city and epidemiological week.
        climate_cov: Covariance matrices per city and epidemiological week (long format).
        start_years: Years whose weeks are used as forecast starts (e.g. the hold-out test years).
        model_definition: Registered MLflow model (e.g. "next_week_prediction/latest").
        horizon: Number of weeks forecast from every start.
        start_every: Use every n-th week of ``start_years`` as a start.
        n_samples: Number of Monte Carlo samples per forecast.
        include_model_error: Whether to add the model error drawn from the model's error model.
        output_dir: Directory for the CSV with the coverage per horizon.
        seed: Seed of the random number generator.

    Returns:
        DataFrame with one row per horizon and one column per interval.
    """
    model, feature_columns, n_lags, error_model = load_model(model_definition)
    if not include_model_error:
        error_model = None
    rng = np.random.default_rng(seed)

    records = []
    for city in City:
        city_df = df_features[df_features["city_name"] == city.value].sort_values("date")
        observed = city_df.set_index("date")["log_p_inc100k"]
        starts = city_df[city_df["year"].isin(start_years)].iloc[::start_every]
        for _, start in starts.iterrows():
            history = get_history(city_df, start["date"], n_lags)
            if any(np.isnan(values).any() for values in history.values()):
                continue
            dates, paths = predict_trajectories(
                model=model,
                feature_columns=feature_columns,
                city=city,
                start_date=start["date"],
                history=history,
                climate_mean=climate_mean,
                climate_cov=climate_cov,
                n_steps=horizon,
                n_samples=n_samples,
                rng=rng,
                error_model=error_model,
                population=start["pop"],
            )
            truth = observed.reindex(dates).to_numpy()
            for interval in INTERVALS:
                lower, upper = np.percentile(paths, [50 * (1 - interval), 50 * (1 + interval)], axis=0)
                for h in np.flatnonzero(~np.isnan(truth)):
                    records.append({
                        "horizon_weeks": h + 1,
                        "interval": interval,
                        "inside": lower[h] <= truth[h] <= upper[h],
                    })

    results = pd.DataFrame(records)
    coverage = results.pivot_table(index="horizon_weeks", columns="interval", values="inside")
    overall = results.groupby("interval")["inside"].mean()
    n_weeks = len(results) // len(INTERVALS)
    print(f"Coverage of the central prediction intervals ({n_weeks} forecast weeks):")
    print("  " + ", ".join(f"{interval:.0%} interval: {share:.1%}" for interval, share in overall.items()))
    print(coverage.iloc[[0, 3, 7, 12, horizon - 1]].round(2).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "calibration.csv"
    coverage.to_csv(csv_path)
    print(f"Coverage per horizon saved to: {csv_path}")
    return coverage


if __name__ == "__main__":
    from model_synthesis.backtesting import make_backtest_splits
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
    splits = make_backtest_splits(df_features["year"].unique(), **TRAIN_SETTINGS["backtesting"])

    main(
        df_features=df_features,
        climate_mean=climate_dict["mean"],
        climate_cov=climate_dict["cov"],
        start_years=splits.test_years,
        model_definition=MODEL_DEFINITION,
    )
