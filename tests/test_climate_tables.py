import numpy as np
import pytest

from utils.climate_tables import MAX_EPI_WEEK, create_climate_tables, table_to_matrix
from utils.constants import CLIMATE_VARS
from utils.feature_extraction import prepare_weekly_data


def test_every_city_and_week_gets_a_valid_covariance_matrix(raw_df):
    weekly = prepare_weekly_data(raw_df, CLIMATE_VARS)

    tables = create_climate_tables(weekly, CLIMATE_VARS, window=2)

    assert len(tables["mean"]) == 2 * MAX_EPI_WEEK
    cov = table_to_matrix(tables["cov"], "Vitória", 53, CLIMATE_VARS)
    assert list(cov.index) == CLIMATE_VARS
    assert list(cov.columns) == CLIMATE_VARS
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0


# without pooling, week 53 has a single observation and no covariance
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_window_pools_weeks_across_the_turn_of_the_year(raw_df):
    weekly = prepare_weekly_data(raw_df, CLIMATE_VARS)
    weekly["tempmed"] = np.where(weekly["epi_week"] >= 52, 100.0, 0.0)

    def mean_of_week_1(window):
        means = create_climate_tables(weekly, CLIMATE_VARS, window=window)["mean"]
        return means.loc[means["epi_week"] == 1, "tempmed"]

    assert (mean_of_week_1(window=0) == 0.0).all()
    assert (mean_of_week_1(window=2) > 0.0).all()
