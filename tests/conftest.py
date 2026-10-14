import numpy as np
import pandas as pd
import pytest

from utils.constants import CLIMATE_VARS

# Epidemiological weeks 2020-01 to 2022-52 (2020 has 53 weeks); week 2020-01 starts on 2019-12-29
EPI_WEEKS = [(2020, week) for week in range(1, 54)] + [
    (year, week) for year in (2021, 2022) for week in range(1, 53)
]
FIRST_WEEK_START = pd.Timestamp("2019-12-29")


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic data in the raw InfoDengue layout: two cities, 157 weeks, rows in random order."""
    rng = np.random.default_rng(0)
    n_weeks = len(EPI_WEEKS)
    week_starts = FIRST_WEEK_START + pd.to_timedelta(7 * np.arange(n_weeks), unit="D")
    frames = []
    for city in ("Rio de Janeiro", "Vitória"):
        frame = pd.DataFrame({
            "city_name": city,
            "SE": [100 * year + week for year, week in EPI_WEEKS],
            "data_iniSE": week_starts.strftime("%Y-%m-%d"),
            "p_inc100k": np.exp(rng.normal(2.0, 1.0, n_weeks)),
            "pop": 1_000_000.0,
        })
        for i, var in enumerate(CLIMATE_VARS):
            frame[var] = 20.0 + i + rng.normal(0.0, 1.0, n_weeks)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=0)
