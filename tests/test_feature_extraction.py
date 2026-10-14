import numpy as np
import pandas as pd
import pytest

from utils.constants import CLIMATE_VARS
from utils.feature_extraction import (
    TARGET_COLUMN,
    build_feature_columns,
    prepare_weekly_data,
    run_feature_pipeline,
)


def test_prepare_weekly_data_sorts_weeks_and_derives_calendar(raw_df):
    weekly = prepare_weekly_data(raw_df, CLIMATE_VARS)

    assert weekly.groupby("city_name")["date"].apply(lambda d: d.is_monotonic_increasing).all()
    first = weekly.iloc[0]
    assert (first["year"], first["epi_week"]) == (2020, 1)
    assert first["date"] == pd.Timestamp("2019-12-29")
    assert first["city_encoded"] == 1  # Rio de Janeiro


def test_misdated_week_is_repaired_from_epi_week(raw_df, caplog):
    expected = prepare_weekly_data(raw_df, CLIMATE_VARS)["date"]
    raw_df.loc[raw_df["SE"] == 202015, "data_iniSE"] = "2020-04-08"  # a Wednesday

    weekly = prepare_weekly_data(raw_df, CLIMATE_VARS)

    pd.testing.assert_series_equal(weekly["date"], expected)
    assert "inconsistent" in caplog.text


def test_missing_week_raises(raw_df):
    with pytest.raises(ValueError, match="gaps"):
        prepare_weekly_data(raw_df[raw_df["SE"] != 202110], CLIMATE_VARS)


def test_unknown_city_raises(raw_df):
    raw_df["city_name"] = raw_df["city_name"].replace("Vitória", "Recife")

    with pytest.raises(ValueError, match="Recife"):
        prepare_weekly_data(raw_df, CLIMATE_VARS)


def test_zero_incidence_is_treated_as_missing(raw_df):
    raw_df.loc[raw_df["SE"].isin([202101, 202102]), "p_inc100k"] = 0.0

    weekly = prepare_weekly_data(raw_df, CLIMATE_VARS)

    assert weekly["log_p_inc100k"].isna().sum() == 4  # two weeks in each of the two cities
    assert np.isfinite(weekly["log_p_inc100k"].dropna()).all()


def test_build_feature_columns():
    columns = build_feature_columns(["tempmed", "log_p_inc100k", "epi_week"], n_lags=1)

    assert columns == [
        "tempmed_(t+1)", "tempmed_(t)", "tempmed_(t-1)",
        "log_p_inc100k_(t)", "log_p_inc100k_(t-1)",
        "epi_week",
    ]


def test_without_lags_only_week_t_and_next_week_weather_are_used():
    columns = build_feature_columns(["tempmed", "log_p_inc100k", "epi_week"], n_lags=0)

    assert columns == ["tempmed_(t+1)", "tempmed_(t)", "log_p_inc100k_(t)", "epi_week"]


def test_lagged_features_are_shifted_within_each_city(raw_df):
    features = run_feature_pipeline(raw_df, CLIMATE_VARS, n_lags=1)

    for _, city_df in features.groupby("city_name"):
        values = city_df["tempmed"].to_numpy()
        np.testing.assert_array_equal(city_df["tempmed_(t+1)"].to_numpy()[:-1], values[1:])
        np.testing.assert_array_equal(city_df["tempmed_(t-1)"].to_numpy()[1:], values[:-1])
        log_inc = city_df["log_p_inc100k"].to_numpy()
        np.testing.assert_array_equal(city_df["log_p_inc100k_(t+1)"].to_numpy()[:-1], log_inc[1:])
        # no values leak across the boundary between two cities
        assert np.isnan(city_df["tempmed_(t+1)"].iloc[-1])
        assert np.isnan(city_df["tempmed_(t-1)"].iloc[0])


def test_target_is_the_weekly_change_and_belongs_to_the_predicted_year(raw_df):
    features = run_feature_pipeline(raw_df, CLIMATE_VARS, n_lags=0)
    city_df = features[features["city_name"] == "Vitória"]

    log_inc = city_df["log_p_inc100k"].to_numpy()
    np.testing.assert_allclose(city_df[TARGET_COLUMN].to_numpy()[:-1], np.diff(log_inc))
    last_week_of_2020 = city_df[(city_df["year"] == 2020) & (city_df["epi_week"] == 53)].iloc[0]
    assert last_week_of_2020["target_year"] == 2021


@pytest.mark.parametrize("n_lags", [-1, 1.5])
def test_invalid_n_lags_raises(raw_df, n_lags):
    with pytest.raises(ValueError):
        run_feature_pipeline(raw_df, CLIMATE_VARS, n_lags=n_lags)
