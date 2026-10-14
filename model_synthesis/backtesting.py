"""Backtesting (rolling-origin evaluation) of the next-week model, scored against persistence."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.feature_extraction import TARGET_COLUMN


def persistence_skill(delta_true, delta_pred) -> float:
    """Skill score relative to the persistence forecast x_(t+1) = x_(t).

    With the weekly change delta = x_(t+1) - x_(t), persistence predicts delta = 0, so

        skill = 1 - sum((delta - delta_pred)^2) / sum(delta^2)

    1 is a perfect forecast, 0 is as good as persistence and negative values are worse.
    """
    delta_true = np.asarray(delta_true, dtype=float)
    delta_pred = np.asarray(delta_pred, dtype=float)
    return float(1.0 - np.sum((delta_true - delta_pred) ** 2) / np.sum(delta_true**2))


@dataclass(frozen=True)
class Fold:
    train_years: list[int]
    validation_years: list[int]


@dataclass(frozen=True)
class BacktestSplits:
    folds: list[Fold]
    development_years: list[int]  # all years before the hold-out test years
    test_years: list[int]


def make_backtest_splits(
    years: Sequence[int], n_test_years: int, n_validation_years: int, n_folds: int
) -> BacktestSplits:
    """Expanding-window backtesting splits by year.

    The last ``n_test_years`` years are held out for the final test. Before them, ``n_folds``
    consecutive windows of ``n_validation_years`` years are each validated with a model
    trained on all earlier years. For 2010-2025 and (2, 2, 4):

        train 2010-2015 -> validate 2016-2017
        train 2010-2017 -> validate 2018-2019
        train 2010-2019 -> validate 2020-2021
        train 2010-2021 -> validate 2022-2023
        train 2010-2023 -> test     2024-2025

    Args:
        years: Years present in the data.
        n_test_years: Number of last years held out for testing.
        n_validation_years: Years ahead validated in each fold.
        n_folds: Number of backtesting folds.

    Returns:
        BacktestSplits with the folds in chronological order.
    """
    if min(n_test_years, n_validation_years, n_folds) < 1:
        raise ValueError("n_test_years, n_validation_years and n_folds must be positive")
    years = sorted({int(year) for year in years})
    development_years, test_years = years[:-n_test_years], years[-n_test_years:]
    first_validation = len(development_years) - n_folds * n_validation_years
    if first_validation < 1:
        raise ValueError(
            f"{len(years)} years are too few for {n_folds} folds of {n_validation_years} "
            f"validation years plus {n_test_years} test years"
        )
    folds = [
        Fold(
            train_years=development_years[:start],
            validation_years=development_years[start : start + n_validation_years],
        )
        for start in range(first_validation, len(development_years), n_validation_years)
    ]
    return BacktestSplits(folds=folds, development_years=development_years, test_years=test_years)


def rows_for_years(df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """Rows whose predicted week falls into ``years``."""
    return df[df["target_year"].isin(years)]


def backtest(
    make_model: Callable[[], object],
    df: pd.DataFrame,
    feature_columns: list[str],
    folds: Sequence[Fold],
) -> dict:
    """Fit a new model in every fold and score its predictions of the fold's validation years.

    Args:
        make_model: Returns an unfitted model with .fit() and .predict().
        df: Feature table without missing values in ``feature_columns`` and the target.
        feature_columns: Input columns of the model.
        folds: Backtesting folds.

    Returns:
        Dict with the persistence skill of all out-of-fold predictions together ("skill"),
        the skill of every fold ("fold_skills") and the out-of-fold residuals ("residuals",
        a Series indexed like ``df``).
    """
    fold_skills, residuals, sse_persistence = [], [], 0.0
    for fold in folds:
        train = rows_for_years(df, fold.train_years)
        validation = rows_for_years(df, fold.validation_years)
        model = make_model().fit(train[feature_columns], train[TARGET_COLUMN])
        delta_true = validation[TARGET_COLUMN]
        delta_pred = model.predict(validation[feature_columns])

        fold_skills.append(persistence_skill(delta_true, delta_pred))
        residuals.append(delta_true - delta_pred)
        sse_persistence += np.sum(delta_true**2)

    residuals = pd.concat(residuals)
    return {
        "skill": float(1.0 - np.sum(residuals**2) / sse_persistence),
        "fold_skills": fold_skills,
        "residuals": residuals,
    }
