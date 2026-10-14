"Compute the weekly climate tables from raw data and write them to parquet files"
import pandas as pd

from utils.config import PROJECT_ROOT
from utils.feature_extraction import prepare_weekly_data
from utils.climate_tables import create_climate_tables
from utils.constants import CLIMATE_VARS


def write_climate_tables(dataset: str, columns: list[str]) -> None:
    """Compute mean, correlation and covariance of ``columns`` per city and epidemiological week, and save them.

    Args:
        dataset: Name of the raw dataset (without extension).
        columns: Climate variables.
    """
    df_raw = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / f"{dataset}.parquet")

    df_weekly = prepare_weekly_data(df_raw, columns=columns)

    climate_dict = create_climate_tables(df_weekly=df_weekly, columns=columns)

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    climate_dict["mean"].to_parquet(output_dir / "climate_tables_mean.parquet", index=False)
    climate_dict["corr"].to_parquet(output_dir / "climate_tables_correlation.parquet", index=False)
    climate_dict["cov"].to_parquet(output_dir / "climate_tables_covariance.parquet", index=False)


if __name__ == "__main__":

    DATA_SET = "infodengue_capitals_subsetBR"
    write_climate_tables(dataset=DATA_SET, columns=CLIMATE_VARS)
