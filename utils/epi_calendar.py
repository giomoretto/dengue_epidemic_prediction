"""Epidemiological week calendar used by InfoDengue (weeks run Sunday to Saturday)."""

import numpy as np
import pandas as pd

ONE_WEEK = np.timedelta64(7, "D")


def epi_week_start(year, week) -> pd.DatetimeIndex:
    """First day (a Sunday) of epidemiological week ``week`` of ``year``.

    Week 1 is the week that contains 4 January, i.e. the first week with at least
    four days in the new year. Accepts scalars or array-likes.
    """
    year = np.atleast_1d(np.asarray(year, dtype="int64"))
    week = np.asarray(week, dtype="int64")
    jan4 = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]") + np.timedelta64(3, "D")
    days_after_sunday = (jan4.view("int64") + 4) % 7  # day 0 (1970-01-01) was a Thursday
    week1_start = jan4 - days_after_sunday.astype("timedelta64[D]")
    return pd.DatetimeIndex(week1_start + (7 * (week - 1)).astype("timedelta64[D]"))


def epi_week_from_date(dates) -> tuple[np.ndarray, np.ndarray]:
    """Epidemiological (year, week) of week start dates; inverse of ``epi_week_start``."""
    dates = pd.DatetimeIndex(np.atleast_1d(dates))
    years = (dates + np.timedelta64(3, "D")).year.to_numpy()  # a week belongs to the year of its Wednesday
    weeks = (dates - epi_week_start(years, 1)).days.to_numpy() // 7 + 1
    return years, weeks
