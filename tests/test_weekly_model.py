import numpy as np
import pandas as pd
import pytest

from model_definition.error_model import ErrorModel
from model_definition.weekly_model import (
    build_features,
    get_climate_sample,
    get_history,
    predict_trajectories,
)
from utils.climate_tables import create_climate_tables
from utils.constants import CLIMATE_VARS
from utils.epi_calendar import ONE_WEEK
from utils.feature_extraction import (
    TARGET_VARIABLE,
    build_feature_columns,
    lagged_name,
    prepare_weekly_data,
    run_feature_pipeline,
)
from utils.types import City


class GrowthModel:
    """Stand-in for the model of the weekly change: log incidence grows by 0.1 per week."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.1)


def test_simulation_features_match_training_features(raw_df):
    """Inference must build exactly the feature vector the model was trained on."""
    n_lags = 2
    features = run_feature_pipeline(raw_df, CLIMATE_VARS, n_lags=n_lags)
    feature_columns = build_feature_columns(
        [*CLIMATE_VARS, "city_encoded", "epi_week", TARGET_VARIABLE], n_lags
    )
    city_df = features[features["city_name"] == City.VITORIA.value]
    row = city_df.iloc[40]

    history = get_history(city_df, row["date"], n_lags)
    climate_next = row[[lagged_name(var, 1) for var in CLIMATE_VARS]].to_numpy(dtype=float)
    simulated = build_features(
        {var: values[None, :] for var, values in history.items()},
        climate_next[None, :],
        City.VITORIA,
        row["epi_week"],
    )

    np.testing.assert_allclose(
        simulated[feature_columns].iloc[0].to_numpy(dtype=float),
        row[feature_columns].to_numpy(dtype=float),
    )


def test_climate_sample_keeps_the_order_of_the_variables(raw_df):
    weekly = prepare_weekly_data(raw_df, CLIMATE_VARS)
    rng = np.random.default_rng(1)
    for i, var in enumerate(CLIMATE_VARS):
        weekly[var] = rng.normal(0.0, 2.0**i, len(weekly))  # std. dev. doubles from one variable to the next
    tables = create_climate_tables(weekly, CLIMATE_VARS, window=10)

    sample = get_climate_sample(City.VITORIA, 30, tables["mean"], tables["cov"], size=20_000, rng=rng)

    assert np.all(np.diff(sample.std(axis=0)) > 0)


def _simulate(raw_df, n_samples=4, **kwargs):
    weekly = prepare_weekly_data(raw_df, CLIMATE_VARS)
    tables = create_climate_tables(weekly, CLIMATE_VARS)
    city_df = weekly[weekly["city_name"] == City.RIO_DE_JANEIRO.value]
    start = city_df.iloc[50]
    dates, log_inc = predict_trajectories(
        model=GrowthModel(),
        feature_columns=["log_p_inc100k_(t)", "tempmed_(t+1)", "epi_week"],
        city=City.RIO_DE_JANEIRO,
        start_date=start["date"],
        history=get_history(city_df, start["date"], n_lags=0),
        climate_mean=tables["mean"],
        climate_cov=tables["cov"],
        n_steps=10,
        n_samples=n_samples,
        **kwargs,
    )
    return start, dates, log_inc


def test_trajectories_feed_predictions_back_as_state(raw_df):
    start, dates, log_inc = _simulate(raw_df, rng=np.random.default_rng(0))

    expected = start["log_p_inc100k"] + 0.1 * np.arange(1, 11)
    np.testing.assert_allclose(log_inc, np.tile(expected, (4, 1)))
    assert dates[0] == start["date"] + ONE_WEEK
    assert len(dates) == 10


def test_trajectories_are_reproducible_with_a_seed(raw_df):
    _, _, first = _simulate(raw_df, rng=np.random.default_rng(7), initial_sigma=0.1)
    _, _, second = _simulate(raw_df, rng=np.random.default_rng(7), initial_sigma=0.1)

    np.testing.assert_array_equal(first, second)
    assert np.unique(first[:, 0]).size == 4  # initial noise differs between samples


def _deviation_from_model_path(raw_df, error_model, n_samples=4_000):
    """Simulated minus noise-free trajectories (start value + 0.1 per week), shape (n_samples, 10)."""
    start, _, log_inc = _simulate(
        raw_df, rng=np.random.default_rng(0), error_model=error_model, population=1_000_000, n_samples=n_samples
    )
    return log_inc - (start["log_p_inc100k"] + 0.1 * np.arange(1, 11))


def test_without_noise_the_error_model_changes_nothing(raw_df):
    deviation = _deviation_from_model_path(raw_df, ErrorModel(sigma_eta=0.0, phi=0.0, tau2=0.0), n_samples=4)

    np.testing.assert_allclose(deviation, 0.0, atol=1e-12)


def test_process_error_accumulates_over_the_horizon(raw_df):
    deviation = _deviation_from_model_path(raw_df, ErrorModel(sigma_eta=0.2, phi=0.0, tau2=0.0))

    spread = deviation.std(axis=0)
    assert spread[0] == pytest.approx(0.2, rel=0.05)
    assert spread[9] == pytest.approx(0.2 * np.sqrt(10), rel=0.05)  # random walk of independent errors


def test_counting_noise_is_not_propagated(raw_df):
    deviation = _deviation_from_model_path(raw_df, ErrorModel(sigma_eta=0.0, phi=0.0, tau2=1.0))

    spread = deviation.std(axis=0)
    assert spread[9] <= spread[0]  # noise of the start and of the reported week only, no accumulation


def test_model_error_needs_the_population(raw_df):
    error_model = ErrorModel(sigma_eta=0.2, phi=0.0, tau2=1.0)

    with pytest.raises(ValueError, match="population"):
        _simulate(raw_df, rng=np.random.default_rng(0), error_model=error_model)
