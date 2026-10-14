"""Weekly feature engineering for the next-week incidence model.

Every row of the feature table describes one city in one epidemiological week ``t``.
Column suffixes give the time of a value relative to that week: ``(t+1)`` is the
following week, ``(t)`` the current week and ``(t-k)`` the week k weeks earlier.
The model predicts the weekly change of the log incidence x = log_p_inc100k,

    x_(t+1) = x_(t) + f(epi_week, city, x_(t), ..., x_(t-n_lags), climate_(t+1), climate_(t), ..., climate_(t-n_lags))
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from utils.epi_calendar import ONE_WEEK, epi_week_start
from utils.types import CITY_ENCODING

logger = logging.getLogger(__name__)

TARGET_VARIABLE = "log_p_inc100k"
# Features used without a time suffix
STATIC_FEATURES = ("city_encoded", "epi_week")


def lagged_name(column: str, offset: int) -> str:
    """Name of ``column`` at week ``t + offset``, e.g. tempmed_(t+1), tempmed_(t), tempmed_(t-2)."""
    return f"{column}_(t)" if offset == 0 else f"{column}_(t{offset:+d})"


CURRENT_LEVEL = lagged_name(TARGET_VARIABLE, 0)  # x_(t)
NEXT_LEVEL = lagged_name(TARGET_VARIABLE, 1)  # x_(t+1)
# Model target: weekly change x_(t+1) - x_(t)
TARGET_COLUMN = lagged_name(f"delta_{TARGET_VARIABLE}", 1)


def weekly_cases(log_incidence, population):
    """Weekly cases from the log incidence per 100 000 inhabitants."""
    return np.exp(log_incidence) * population / 100_000


def build_feature_columns(features: Sequence[str], n_lags: int) -> list[str]:
    """Expand base feature names into the model's input columns.

    Climate variables are used at (t+1), (t), ..., (t-n_lags), the autoregressive
    variable (log_p_inc100k) at (t), ..., (t-n_lags) and static features as-is.

    Args:
        features: Base feature names (e.g. CLIMATE_VARS + ["city_encoded", "epi_week", "log_p_inc100k"]).
        n_lags: Number of lagged weeks (t-1, ..., t-n_lags) used in addition to week t.

    Returns:
        List of feature column names ready for model training.
    """
    columns = []
    for feature in features:
        if feature in STATIC_FEATURES:
            columns.append(feature)
        else:
            first_offset = 0 if feature == TARGET_VARIABLE else 1
            columns.extend(lagged_name(feature, offset) for offset in range(first_offset, -n_lags - 1, -1))
    return columns


def prepare_weekly_data(df_raw: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Turn the raw InfoDengue records into a clean, regular weekly panel.

    - The epidemiological week code ``SE`` (YYYYWW) is the time key. ``data_iniSE`` is
      only cross-checked against it, because it contains a misdated week
      (Wednesday 2018-04-04 instead of Sunday 2018-04-08).
    - Every city must have one row per week without gaps, so that shifting by one row
      within a city is shifting by one week.
    - Zero incidence is treated as missing: in this dataset zero weekly counts only occur
      in reporting gaps (Vitória 2021, last weeks of 2025), not as real absence of cases.

    Args:
        df_raw: Raw records with columns city_name, SE, data_iniSE, pop and ``columns``.
        columns: Measured columns to keep. p_inc100k (the target derives from it) and pop
            (to convert incidence into cases) are always kept.

    Returns:
        DataFrame sorted by city and date with columns city_name, date, year, epi_week,
        city_encoded, the kept columns and log_p_inc100k.
    """
    columns = list(dict.fromkeys([*columns, "p_inc100k", "pop"]))
    df = df_raw[["city_name", "SE", "data_iniSE", *columns]].copy()
    df["year"] = df["SE"] // 100
    df["epi_week"] = df["SE"] % 100
    df["date"] = epi_week_start(df["year"], df["epi_week"])

    n_misdated = int((pd.to_datetime(df["data_iniSE"]) != df["date"]).sum())
    if n_misdated:
        logger.warning("%d rows have a data_iniSE inconsistent with their epidemiological week; using SE.", n_misdated)

    df = df.sort_values(["city_name", "date"]).reset_index(drop=True)
    _check_regular_weeks(df)

    df["p_inc100k"] = df["p_inc100k"].where(df["p_inc100k"] > 0)
    df[TARGET_VARIABLE] = np.log(df["p_inc100k"])

    df["city_encoded"] = df["city_name"].map({city.value: code for city, code in CITY_ENCODING.items()})
    if df["city_encoded"].isna().any():
        unknown = sorted(df.loc[df["city_encoded"].isna(), "city_name"].unique())
        raise ValueError(f"Cities missing in CITY_ENCODING: {unknown}")

    return df[["city_name", "date", "year", "epi_week", "city_encoded", *columns, TARGET_VARIABLE]]


def _check_regular_weeks(df: pd.DataFrame) -> None:
    """Raise if any city's series has a missing or duplicated week."""
    steps = df.groupby("city_name")["date"].diff().dropna()
    irregular = steps.index[steps != ONE_WEEK]
    if len(irregular):
        raise ValueError(
            "Weekly series must not have gaps or duplicates, found at:\n"
            f"{df.loc[irregular, ['city_name', 'date']].head()}"
        )


def add_lagged_features(df: pd.DataFrame, columns: Sequence[str], n_lags: int) -> pd.DataFrame:
    """Add the (t+1), (t), ..., (t-n_lags) versions of ``columns``, shifted within each city.

    Args:
        df: Regular weekly panel as returned by ``prepare_weekly_data``.
        columns: Columns to shift.
        n_lags: Number of lagged weeks (t-1, ..., t-n_lags) in addition to week t.

    Returns:
        ``df`` with the lagged columns appended.
    """
    if not isinstance(n_lags, int) or n_lags < 0:
        raise ValueError(f"n_lags must be a non-negative integer, got {n_lags}")

    by_city = df.groupby("city_name", sort=False)
    shifted = {
        lagged_name(column, offset): by_city[column].shift(-offset)
        for column in columns
        for offset in range(1, -n_lags - 1, -1)
    }
    return pd.concat([df, pd.DataFrame(shifted)], axis=1)


def run_feature_pipeline(df: pd.DataFrame, columns: Sequence[str], n_lags: int) -> pd.DataFrame:
    """Build the weekly feature table: clean weekly panel, lagged columns and model target.

    Args:
        df: Raw InfoDengue records.
        columns: Measured columns to keep and lag (e.g. CLIMATE_VARS + ["p_inc100k", "pop"]).
        n_lags: Number of lagged weeks (t-1, ..., t-n_lags) in addition to week t.

    Returns:
        Weekly panel with the lagged versions of ``columns`` and of log_p_inc100k, the target
        TARGET_COLUMN and the epidemiological year of the predicted week ("target_year").
    """
    weekly = prepare_weekly_data(df, columns)
    features = add_lagged_features(weekly, [*columns, TARGET_VARIABLE], n_lags)
    features[TARGET_COLUMN] = features[NEXT_LEVEL] - features[CURRENT_LEVEL]
    # Train/validation/test splits assign each row to the year of the week it predicts
    features["target_year"] = features.groupby("city_name", sort=False)["year"].shift(-1)
    return features
