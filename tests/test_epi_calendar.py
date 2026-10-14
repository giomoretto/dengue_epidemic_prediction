import pandas as pd
import pytest

from utils.epi_calendar import epi_week_from_date, epi_week_start


@pytest.mark.parametrize(
    "year, week, expected",
    [
        (2010, 1, "2010-01-03"),  # first week of the dataset
        (2018, 15, "2018-04-08"),  # misdated as 2018-04-04 in the raw data
        (2020, 53, "2020-12-27"),
        (2025, 1, "2024-12-29"),  # week 1 can start in the previous calendar year
        (2025, 53, "2025-12-28"),  # last week of the dataset
    ],
)
def test_epi_week_start_matches_known_dates(year, week, expected):
    assert epi_week_start(year, week)[0] == pd.Timestamp(expected)


def test_epi_week_from_date_inverts_epi_week_start():
    sundays = pd.date_range("2009-12-27", "2026-12-27", freq="W-SUN")

    years, weeks = epi_week_from_date(sundays)

    assert (epi_week_start(years, weeks) == sundays).all()
    assert weeks.min() == 1
    assert weeks.max() == 53
