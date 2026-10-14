import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from model_synthesis.backtesting import (
    Fold,
    backtest,
    make_backtest_splits,
    persistence_skill,
    rows_for_years,
)
from utils.constants import CLIMATE_VARS
from utils.feature_extraction import TARGET_COLUMN, build_feature_columns, run_feature_pipeline


def test_persistence_skill():
    delta = np.array([0.5, -0.5, 1.0])

    assert persistence_skill(delta, delta) == 1.0
    assert persistence_skill(delta, np.zeros(3)) == 0.0  # persistence predicts no change
    assert persistence_skill(delta, 2 * delta) == 0.0
    assert persistence_skill(delta, -delta) == -3.0


def test_backtest_splits_expand_the_training_window():
    splits = make_backtest_splits(range(2010, 2026), n_test_years=2, n_validation_years=2, n_folds=4)

    assert splits.test_years == [2024, 2025]
    assert splits.development_years == list(range(2010, 2024))
    assert splits.folds == [
        Fold(train_years=list(range(2010, 2016)), validation_years=[2016, 2017]),
        Fold(train_years=list(range(2010, 2018)), validation_years=[2018, 2019]),
        Fold(train_years=list(range(2010, 2020)), validation_years=[2020, 2021]),
        Fold(train_years=list(range(2010, 2022)), validation_years=[2022, 2023]),
    ]


def test_too_few_years_for_the_backtest_raise():
    with pytest.raises(ValueError, match="too few"):
        make_backtest_splits(range(2010, 2020), n_test_years=2, n_validation_years=2, n_folds=4)


def test_backtest_scores_every_fold_on_unseen_years(raw_df):
    features = run_feature_pipeline(raw_df, CLIMATE_VARS, n_lags=1)
    feature_columns = build_feature_columns([*CLIMATE_VARS, "epi_week", "log_p_inc100k"], n_lags=1)
    df = features.dropna(subset=[*feature_columns, TARGET_COLUMN])
    splits = make_backtest_splits(df["target_year"].unique(), n_test_years=1, n_validation_years=1, n_folds=1)

    result = backtest(lambda: DummyRegressor(strategy="constant", constant=0.0), df, feature_columns, splits.folds)

    assert result["fold_skills"] == [0.0]  # a constant zero change is the persistence forecast
    assert result["skill"] == 0.0
    assert result["residuals"].index.equals(rows_for_years(df, [2021]).index)
    assert rows_for_years(df, splits.folds[0].train_years)["target_year"].max() == 2020
