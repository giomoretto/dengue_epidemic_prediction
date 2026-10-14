"""Monte-Carlo simulation of next-year weekly incidence for all cities from several start weeks."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_definition.weekly_model import get_history, load_model, predict_trajectories
from utils.config import PROJECT_ROOT
from utils.types import City

FIG_SIZE = (10, 8)
N_STEPS = 52  # one year of weekly predictions
INITIAL_SIGMA = 0.1  # 10% (log-normal) uncertainty of p_inc100k in the start week
OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction"

# Categorical colors in fixed order (one per start point), neutral ink for observations
START_POINT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OBSERVED_COLOR = "#52514e"
GRID_COLOR = "#e1e0d9"


def main(
    df_features: pd.DataFrame,
    climate_mean: pd.DataFrame,
    climate_cov: pd.DataFrame,
    start_points: list[tuple[int, int]],
    n_samples: int,
    uncertainty_in_start: bool,
    model_definition: str,
    include_model_error: bool = True,
    output_dir: Path = OUTPUT_DIR,
    seed: int = 42,
):
    """Monte-carlo sampling of next-year predictions for all cities, starting from specified weeks.

    Args:
        df_features: Weekly feature table (one row per city and week).
        climate_mean: Mean climate per city and epidemiological week.
        climate_cov: Covariance matrices per city and epidemiological week (long format).
        start_points: (year, epidemiological week) of the last observed week of each simulation.
        n_samples: Number of Monte Carlo samples.
        uncertainty_in_start: Whether to add noise to the initial p_inc100k value.
        model_definition: Registered MLflow model (e.g. "next_week_prediction/latest").
        include_model_error: Whether to add the model error drawn from the model's error model.
        output_dir: Directory for the figures and the CSV of weekly case predictions.
        seed: Seed of the random number generator.
    """
    if len(start_points) > len(START_POINT_COLORS):
        raise ValueError(f"At most {len(START_POINT_COLORS)} start points can be told apart")

    model, feature_columns, n_lags, error_model = load_model(model_definition)
    if include_model_error and error_model is None:
        print("Warning: The model was logged without error model; simulating without model error.")
    if not include_model_error:
        error_model = None
    rng = np.random.default_rng(seed)
    initial_sigma = INITIAL_SIGMA if uncertainty_in_start else 0.0

    # Two figures: log scale (geometric mean) and natural scale (arithmetic mean)
    fig_log, axes_log = plt.subplots(len(City), 1, figsize=FIG_SIZE, sharex=True)
    fig_nat, axes_nat = plt.subplots(len(City), 1, figsize=FIG_SIZE, sharex=True)

    # Collect weekly case estimates for CSV output
    case_tables = []

    for ax_log, ax_nat, city in zip(axes_log, axes_nat, City):
        city_df = df_features[df_features["city_name"] == city.value].sort_values("date")

        # Plot full timeline as context
        ax_log.plot(city_df["date"], city_df["log_p_inc100k"], label="Observed",
                    color=OBSERVED_COLOR, linewidth=0.8)
        ax_nat.plot(city_df["date"], city_df["p_inc100k"], label="Observed",
                    color=OBSERVED_COLOR, linewidth=0.8)

        for color, (start_year, start_week) in zip(START_POINT_COLORS, start_points):
            start_row = city_df[(city_df["year"] == start_year) & (city_df["epi_week"] == start_week)]
            if start_row.empty:
                print(f"Warning: No data found for {city.value}, {start_year}W{start_week}. Skipping.")
                continue
            start_date = start_row["date"].iloc[0]
            history = get_history(city_df, start_date, n_lags)
            if any(np.isnan(values).any() for values in history.values()):
                print(f"Warning: Missing observations up to {city.value}, {start_year}W{start_week}. Skipping.")
                continue

            pop = start_row["pop"].iloc[0]
            dates, log_paths = predict_trajectories(
                model=model,
                feature_columns=feature_columns,
                city=city,
                start_date=start_date,
                history=history,
                climate_mean=climate_mean,
                climate_cov=climate_cov,
                n_steps=N_STEPS,
                n_samples=n_samples,
                rng=rng,
                initial_sigma=initial_sigma,
                error_model=error_model,
                population=pop,
            )
            paths = np.exp(log_paths)  # (n_samples, N_STEPS)

            # Geometric mean (mean in log space) and arithmetic mean (mean in natural space),
            # each with the 95% interval of the samples
            label = f"{start_year}W{start_week:02d}"
            plot_dates = dates.insert(0, start_date)
            log_start = history["log_p_inc100k"][0]
            _plot_forecast(ax_log, plot_dates, log_start, log_paths.mean(axis=0),
                           np.percentile(log_paths, [2.5, 97.5], axis=0), color, label)
            _plot_forecast(ax_nat, plot_dates, np.exp(log_start), paths.mean(axis=0),
                           np.percentile(paths, [2.5, 97.5], axis=0), color, label)

            # Convert p_inc100k to cases: cases = p_inc100k * pop / 100_000
            cases = paths * pop / 100_000
            observed = city_df.set_index("date")["p_inc100k"].reindex(dates) * pop / 100_000
            ci_lower, ci_upper = np.percentile(cases, [2.5, 97.5], axis=0)
            case_tables.append(pd.DataFrame({
                "city": city.value,
                "start_year": start_year,
                "start_week": start_week,
                "week_start": dates,
                "horizon_weeks": np.arange(1, N_STEPS + 1),
                "observed_cases": observed.round().to_numpy(),
                "predicted_cases": cases.mean(axis=0).round(),
                "ci_lower": ci_lower.round(),
                "ci_upper": ci_upper.round(),
            }))

        ax_log.set_ylabel("log(p_inc100k)")
        ax_nat.set_ylabel("p_inc100k")
        for ax in (ax_log, ax_nat):
            ax.set_title(city.value)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, color=GRID_COLOR, linewidth=0.5)

    for ax in [*axes_log, *axes_nat]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes_log[-1].set_xlabel("Year")
    axes_nat[-1].set_xlabel("Year")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write weekly case estimates to CSV
    if case_tables:
        cases_df = pd.concat(case_tables, ignore_index=True)
        csv_path = output_dir / "predicted_weekly_cases.csv"
        cases_df.to_csv(csv_path, index=False)
        print(f"Weekly cases CSV saved to: {csv_path}")

        # Calibration check: share of observed weeks inside the 95% interval (nominal: 95%)
        observed_weeks = cases_df.dropna(subset=["observed_cases"])
        inside = observed_weeks["observed_cases"].between(observed_weeks["ci_lower"], observed_weeks["ci_upper"])
        print(f"Observed weeks inside the 95% interval: {inside.mean():.0%} of {len(observed_weeks)}")

    error_label = "with model error" if error_model is not None else "without model error"
    fig_log.suptitle(
        f"Monte Carlo Predictions – Geometric Mean (log scale)\n"
        f"(mean and 95% interval, N={n_samples}, σ={initial_sigma}, {error_label})",
        fontsize=14,
    )
    fig_log.tight_layout(rect=[0, 0, 1, 0.96])

    fig_nat.suptitle(
        f"Monte Carlo Predictions – Arithmetic Mean (natural scale)\n"
        f"(mean and 95% interval, N={n_samples}, σ={initial_sigma}, {error_label})",
        fontsize=14,
    )
    fig_nat.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plots
    output_log = output_dir / "monte_carlo_geometric_mean.png"
    output_nat = output_dir / "monte_carlo_arithmetic_mean.png"
    fig_log.savefig(output_log, dpi=300, bbox_inches="tight")
    fig_nat.savefig(output_nat, dpi=300, bbox_inches="tight")
    print(f"Log plot saved to: {output_log}")
    print(f"Natural plot saved to: {output_nat}")

    plt.show()


def _plot_forecast(ax, dates, start_value, center, band, color, label):
    """Plot a forecast line from the observed start value, with its 95% band."""
    center = np.concatenate([[start_value], center])
    lower, upper = (np.concatenate([[start_value], bound]) for bound in band)
    ax.fill_between(dates, lower, upper, color=color, alpha=0.2, linewidth=0)
    ax.plot(dates, center, label=label, color=color, linewidth=1.5)
    ax.plot(dates[0], start_value, marker="s", markersize=5, color=color, zorder=5)


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
        climate_mean=climate_dict["mean"],
        climate_cov=climate_dict["cov"],
        start_points=[(2015, 23), (2017, 36), (2019, 10), (2025, 1)],
        n_samples=500,
        uncertainty_in_start=True,
        model_definition=MODEL_DEFINITION,
    )
