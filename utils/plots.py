import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.epi_calendar import ONE_WEEK
from utils.feature_extraction import CURRENT_LEVEL, NEXT_LEVEL


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    """Create a residual plot (predicted vs residuals)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.5)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    return fig


def plot_predicted_vs_true(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    """Create a predicted vs true scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.5)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs True")
    ax.legend()
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: list[str], title: str = "Feature Importance") -> Figure:
    """Create a horizontal bar chart of feature importances."""
    sorted_idx = importances.argsort()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _binned(cases: np.ndarray, values: np.ndarray, statistic, edges: np.ndarray,
            min_per_bin: int) -> list[tuple[float, float]]:
    """(Geometric mean of the cases, statistic of the values) for every bin with enough rows."""
    bins = np.digitize(cases, edges)
    return [
        (np.exp(np.log(cases[bins == b]).mean()), statistic(values[bins == b]))
        for b in np.unique(bins)
        if np.sum(bins == b) >= min_per_bin
    ]


def plot_error_model(residuals: pd.DataFrame, excluded, pairs: pd.DataFrame, error_model,
                     min_per_bin: int = 20) -> Figure:
    """Plot spread and lag-1 correlation of the residuals against the weekly cases, with the fitted error model.

    Args:
        residuals: Out-of-fold residuals with columns cases (weekly cases in week t) and residual.
        excluded: True for residuals left out of the fit (turn of the year).
        pairs: Consecutive residuals with columns residual, previous and cases.
        error_model: Fitted error model with .change_std(cases) and .change_autocorrelation(cases).
        min_per_bin: Minimum number of residuals per plotted bin.
    """
    cases = np.maximum(residuals["cases"].to_numpy(dtype=float), 1.0)
    values = residuals["residual"].to_numpy(dtype=float)
    excluded = np.asarray(excluded, dtype=bool)
    pair_cases = np.maximum(pairs["cases"].to_numpy(dtype=float), 1.0)
    edges = np.logspace(0, np.log10(cases.max()) + 1e-9, 13)
    grid = np.logspace(0, np.log10(cases.max()), 200)

    fig, (ax_std, ax_corr) = plt.subplots(1, 2, figsize=(13, 5.5))
    for mask, label, color in (
        (~excluded, "Out-of-fold residuals", "#52514e"),
        (excluded, "Epi weeks 51-53 (not fitted)", "#eb6834"),
    ):
        points = _binned(cases[mask], values[mask], np.std, edges, min_per_bin)
        if points:
            ax_std.plot(*zip(*points), "o", color=color, markersize=6, label=label)
    ax_std.plot(grid, error_model.change_std(grid), color="#2a78d6", linewidth=2, label="Error model")
    ax_std.set_ylabel("Std. dev. of the residual of the weekly change")
    ax_std.set_title(f"Spread: σ_η = {error_model.sigma_eta:.3f}, τ² = {error_model.tau2:.2f}")

    points = _binned(pair_cases, pairs[["residual", "previous"]].to_numpy(),
                     lambda pair: np.corrcoef(pair.T)[0, 1], edges, min_per_bin)
    ax_corr.plot(*zip(*points), "o", color="#52514e", markersize=6, label="Out-of-fold residuals")
    ax_corr.plot(grid, error_model.change_autocorrelation(grid), color="#2a78d6", linewidth=2, label="Error model")
    ax_corr.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax_corr.set_ylabel("Correlation with the residual of the previous week")
    ax_corr.set_title(f"Persistence: φ = {error_model.phi:.2f}")

    for ax in (ax_std, ax_corr):
        ax.set_xscale("log")
        ax.set_xlabel("Weekly cases n in week t")
        ax.grid(True, color="#e1e0d9", linewidth=0.5)
        ax.legend()
    fig.suptitle("Error model of the weekly change (binned out-of-fold residuals)")
    fig.tight_layout()
    return fig


def plot_time_series_predictions(
    df: pd.DataFrame,
    model,
    feature_columns: list[str],
    test_years: list[int] | None = None,
) -> Figure:
    """Plot observed vs one-step-ahead predictions of the log incidence for all cities in a 4x1 figure.

    The model predicts the weekly change, so the predicted level is x_(t) + model prediction.

    Args:
        df: Weekly feature table with complete features, must contain 'city_name', 'date',
            'target_year', the current and next log incidence, and all feature columns.
        model: Trained model with a .predict() method.
        feature_columns: List of feature column names used by the model.
        test_years: Hold-out years to shade.

    Returns:
        A matplotlib Figure with 4x1 subplots.
    """
    cities = sorted(df["city_name"].unique())
    fig, axes = plt.subplots(len(cities), 1, figsize=(12, 16), sharex=True)

    for ax, city in zip(axes, cities):
        city_df = df[df["city_name"] == city].set_index("date").sort_index()
        y_pred = city_df[CURRENT_LEVEL] + model.predict(city_df[feature_columns])

        # Regular weekly grid, so that weeks without prediction show as gaps
        weeks = pd.date_range(city_df.index.min(), city_df.index.max(), freq="7D")
        target_dates = weeks + ONE_WEEK
        if test_years:
            test_dates = city_df.index[city_df["target_year"].isin(test_years)] + ONE_WEEK
            ax.axvspan(test_dates.min(), test_dates.max(), color="#e1e0d9", alpha=0.5, label="Hold-out test")
        ax.plot(target_dates, city_df[NEXT_LEVEL].reindex(weeks), label="Observed",
                color="#52514e", linewidth=0.8)
        ax.plot(target_dates, y_pred.reindex(weeks), label="Predicted",
                color="#2a78d6", linewidth=0.8)
        ax.set_ylabel(NEXT_LEVEL)
        ax.set_title(city)
        ax.legend(fontsize=8)
        ax.grid(True, color="#e1e0d9", linewidth=0.5)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Observed vs Predicted by City", fontsize=14)
    fig.tight_layout()
    return fig
