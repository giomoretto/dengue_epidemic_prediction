"""Weekly climatology of each city: mean vector and covariance matrix of the climate
variables per epidemiological week, used to sample weather for the Monte-Carlo simulation."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

MAX_EPI_WEEK = 53


def create_climate_tables(
    df_weekly: pd.DataFrame, columns: Sequence[str], window: int = 2
) -> dict[str, pd.DataFrame]:
    """Compute per-city per-week mean vector, correlation matrix, and covariance matrix of the climate variables.

    The statistics of epidemiological week w pool all observations within ``window`` weeks
    of w (circular over the turn of the year): a single week has only one observation per
    year, too few to estimate a covariance matrix.

    Args:
        df_weekly: Weekly panel with columns city_name, epi_week and ``columns``.
        columns: Climate variables.
        window: Half-width of the pooling window in weeks.

    Returns:
        Dict with "mean" (wide: city_name, epi_week, *columns) and "corr", "cov"
        (long: city_name, epi_week, var_i, var_j, value).
    """
    df_weekly = df_weekly.dropna(subset=columns)

    means, corr_tables, cov_tables = [], [], []
    for city, city_df in df_weekly.groupby("city_name"):
        for week in range(1, MAX_EPI_WEEK + 1):
            distance = (city_df["epi_week"] - week).abs()
            distance = np.minimum(distance, MAX_EPI_WEEK - distance)
            sample = city_df.loc[distance <= window, columns]

            means.append({"city_name": city, "epi_week": week, **sample.mean().to_dict()})
            corr_tables.append(matrix_to_table(sample.corr(), city, week))
            cov_tables.append(matrix_to_table(sample.cov(), city, week))

    return {
        "mean": pd.DataFrame(means),
        "corr": pd.concat(corr_tables, ignore_index=True),
        "cov": pd.concat(cov_tables, ignore_index=True),
    }


def matrix_to_table(matrix: pd.DataFrame, city_name: str, epi_week: int) -> pd.DataFrame:
    """Convert a square matrix to a long table [city_name, epi_week, var_i, var_j, value]."""
    table = matrix.rename_axis(index="var_i", columns="var_j").stack().rename("value").reset_index()
    return table.assign(city_name=city_name, epi_week=epi_week)[
        ["city_name", "epi_week", "var_i", "var_j", "value"]
    ]


def table_to_matrix(
    table: pd.DataFrame, city_name: str, epi_week: int, columns: Sequence[str]
) -> pd.DataFrame:
    """Square matrix of one city and week from a long table, with rows and columns ordered as ``columns``."""
    subset = table[(table["city_name"] == city_name) & (table["epi_week"] == epi_week)]
    # pivot sorts the variables alphabetically; reindex restores the order of the mean vector
    matrix = subset.pivot(index="var_i", columns="var_j", values="value")
    return matrix.reindex(index=columns, columns=columns)
