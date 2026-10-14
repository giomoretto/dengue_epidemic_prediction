"write the weekly feature table as parquet file"
import pandas as pd

from utils.config import PROJECT_ROOT
from utils.feature_extraction import run_feature_pipeline


def write_weekly_features(dataset: str, columns: list[str], n_lags: int) -> pd.DataFrame:
    """Build the weekly feature table from the raw parquet file and write it to data/processed.

    Args:
        dataset (str): Name of the raw dataset (without extension).
        columns (list[str]): Measured columns to keep and lag.
        n_lags (int): Number of lagged weeks in addition to week t.

    Returns:
        pd.DataFrame: The weekly feature table.
    """

    df_raw = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / f"{dataset}.parquet")

    df_features = run_feature_pipeline(df=df_raw, columns=columns, n_lags=n_lags)

    output_path = PROJECT_ROOT / "data" / "processed" / "features_weekly.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)

    return df_features


if __name__ == "__main__":
    from utils.run_complete_pipeline import FEATURE_SETTINGS, FILENAME

    df_features = write_weekly_features(
        dataset=FILENAME, columns=FEATURE_SETTINGS["columns"], n_lags=FEATURE_SETTINGS["n_lags"]
    )
    print(df_features.head())
    print(df_features.columns.sort_values())
