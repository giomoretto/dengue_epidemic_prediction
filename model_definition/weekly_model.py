"""Inference with the next-week incidence model: model loading, feature construction and
Monte-Carlo simulation of weekly trajectories with sampled weather."""

import mlflow
import numpy as np
import pandas as pd

from model_definition.error_model import ErrorModel
from utils.climate_tables import table_to_matrix
from utils.config import MLFLOW_TRACKING_URI
from utils.constants import CLIMATE_VARS
from utils.epi_calendar import epi_week_from_date
from utils.feature_extraction import TARGET_VARIABLE, lagged_name, weekly_cases
from utils.types import CITY_ENCODING, City

# Variables whose recent history is the state of the simulation
STATE_VARIABLES = [*CLIMATE_VARS, TARGET_VARIABLE]
ERROR_MODEL_ARTIFACT = "error_model.json"


def load_model(model_definition: str):
    """Load a registered MLflow model together with its input columns, history length and error model.

    Args:
        model_definition: Registered model name and version (e.g. "next_week_prediction/latest").

    Returns:
        tuple: (model, feature_columns, n_lags, error_model); error_model is None for runs without one.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{model_definition}")
    feature_columns = model.metadata.get_input_schema().input_names()
    run_id = model.metadata.run_id
    n_lags = int(mlflow.get_run(run_id).data.params["n_lags"])

    artifacts = {artifact.path for artifact in mlflow.MlflowClient().list_artifacts(run_id)}
    error_model = None
    if ERROR_MODEL_ARTIFACT in artifacts:
        error_model = ErrorModel.from_dict(mlflow.artifacts.load_dict(f"runs:/{run_id}/{ERROR_MODEL_ARTIFACT}"))
    return model, feature_columns, n_lags, error_model


def get_history(city_df: pd.DataFrame, date: pd.Timestamp, n_lags: int) -> dict[str, np.ndarray]:
    """Observed state at weeks t, t-1, ..., t-n_lags, where t starts on ``date``.

    Args:
        city_df: Weekly panel of one city (columns date and STATE_VARIABLES).
        date: Start date of week t.
        n_lags: Number of lagged weeks in addition to week t.

    Returns:
        Dict variable -> array of length n_lags + 1; element k is the value at week t-k.
    """
    n_weeks = n_lags + 1
    window = city_df.set_index("date").sort_index().loc[:date, STATE_VARIABLES].tail(n_weeks)
    if len(window) < n_weeks or window.index[-1] != date:
        raise ValueError(f"No {n_weeks} weeks of history up to {date.date()}")
    return {var: window[var].to_numpy()[::-1] for var in STATE_VARIABLES}


def get_climate_sample(
    city: City,
    epi_week: int,
    climate_mean: pd.DataFrame,
    climate_cov: pd.DataFrame,
    size: int,
    rng: np.random.Generator,
    randomness: bool = True,
) -> np.ndarray:
    """Draw the weather of one week from the city's weekly climatology.

    Args:
        city: City to sample for.
        epi_week: Epidemiological week (1-53).
        climate_mean: Mean climate per city and week (wide format).
        climate_cov: Covariance matrices per city and week (long format).
        size: Number of samples.
        rng: Random number generator.
        randomness: If True, sample from the multivariate normal; if False, return the mean.

    Returns:
        Array (size, len(CLIMATE_VARS)) with columns in the order of CLIMATE_VARS.
    """
    mean_row = climate_mean[
        (climate_mean["city_name"] == city.value) & (climate_mean["epi_week"] == epi_week)
    ]
    if mean_row.empty:
        raise KeyError(f"No climatology for {city.value}, epidemiological week {epi_week}")
    mean_vector = mean_row[CLIMATE_VARS].to_numpy().ravel()

    if not randomness:
        return np.tile(mean_vector, (size, 1))
    cov_matrix = table_to_matrix(climate_cov, city.value, epi_week, CLIMATE_VARS).to_numpy()
    return rng.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=size)


def build_features(
    history: dict[str, np.ndarray], climate_next: np.ndarray, city: City, epi_week: int
) -> pd.DataFrame:
    """Model inputs for predicting week t+1, named like the columns of the training feature table.

    Args:
        history: Dict variable -> array (n_samples, n_lags + 1); column k holds week t-k.
        climate_next: Weather of week t+1, array (n_samples, len(CLIMATE_VARS)).
        city: City of the samples.
        epi_week: Epidemiological week of t.

    Returns:
        DataFrame with one row per sample.
    """
    columns = {
        lagged_name(var, -k): values[:, k]
        for var, values in history.items()
        for k in range(values.shape[1])
    }
    for i, var in enumerate(CLIMATE_VARS):
        columns[lagged_name(var, 1)] = climate_next[:, i]
    n_samples = len(climate_next)
    columns["epi_week"] = np.full(n_samples, epi_week)
    columns["city_encoded"] = np.full(n_samples, CITY_ENCODING[city])
    return pd.DataFrame(columns)


def predict_trajectories(
    model,
    feature_columns: list[str],
    city: City,
    start_date: pd.Timestamp,
    history: dict[str, np.ndarray],
    climate_mean: pd.DataFrame,
    climate_cov: pd.DataFrame,
    n_steps: int,
    n_samples: int,
    rng: np.random.Generator,
    initial_sigma: float = 0.0,
    randomness: bool = True,
    error_model: ErrorModel | None = None,
    population: float | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Monte-Carlo trajectories of log incidence for the ``n_steps`` weeks after ``start_date``.

    Each step predicts the change from week t to t+1, x_(t+1) = x_(t) + f(...) + eta_(t+1), from
    the state at week t and the weather of week t+1 drawn from the climatology, then feeds the
    result back as the new state (recursive forecasting). With an error model, the state is the
    true log incidence: the counting noise of the observed start value is removed, eta is the
    persistent process error, and the returned reported values carry fresh counting noise that is
    not propagated. All samples are simulated at once.

    Args:
        model: Trained model of the weekly change, with a .predict() method.
        feature_columns: Input columns expected by the model.
        city: City to simulate.
        start_date: Start date of the last observed week t.
        history: Observed state at the start (see ``get_history``).
        climate_mean: Mean climate per city and week (wide format).
        climate_cov: Covariance matrices per city and week (long format).
        n_steps: Number of weeks to predict.
        n_samples: Number of Monte-Carlo samples.
        rng: Random number generator.
        initial_sigma: Std. dev. of the multiplicative (log-normal) noise on the initial incidence.
        randomness: If True, sample the weather; if False, use the climatological mean.
        error_model: Process and counting noise; without it, the model is simulated without error.
        population: Population of the city, converts incidence into the cases that scale the counting noise.

    Returns:
        tuple: (dates of the predicted weeks, array (n_samples, n_steps) of reported log_p_inc100k)
    """
    if error_model is not None and population is None:
        raise ValueError("population is required to scale the counting noise")

    state = {var: np.tile(values, (n_samples, 1)) for var, values in history.items()}
    state[TARGET_VARIABLE][:, 0] += rng.normal(0.0, initial_sigma, n_samples)
    if error_model is not None:
        start = state[TARGET_VARIABLE][:, 0]
        state[TARGET_VARIABLE][:, 0] -= error_model.counting_noise(weekly_cases(start, population), rng)
        eta = error_model.initial_process_error(n_samples, rng)

    dates = start_date + pd.to_timedelta(7 * np.arange(1, n_steps + 1), unit="D")
    _, epi_weeks = epi_week_from_date(dates.insert(0, start_date))

    reported = np.empty((n_samples, n_steps))
    for step in range(n_steps):
        climate_next = get_climate_sample(
            city, epi_weeks[step + 1], climate_mean, climate_cov, n_samples, rng, randomness
        )
        features = build_features(state, climate_next, city, epi_weeks[step])
        log_inc_next = state[TARGET_VARIABLE][:, 0] + model.predict(features[feature_columns])
        if error_model is None:
            reported[:, step] = log_inc_next
        else:
            eta = error_model.next_process_error(eta, rng)
            log_inc_next = log_inc_next + eta
            reported[:, step] = log_inc_next + error_model.counting_noise(
                weekly_cases(log_inc_next, population), rng
            )
        state = _advance(state, climate_next, log_inc_next)
    return dates, reported


def _advance(
    state: dict[str, np.ndarray], climate_next: np.ndarray, log_inc_next: np.ndarray
) -> dict[str, np.ndarray]:
    """Move the state one week forward: week t+1 becomes the new week t."""
    newest = {var: climate_next[:, i] for i, var in enumerate(CLIMATE_VARS)}
    newest[TARGET_VARIABLE] = log_inc_next
    return {var: np.column_stack([newest[var], values[:, :-1]]) for var, values in state.items()}
