"""
Convert CSV data to Parquet format for efficient storage and processing.
"""
import pandas as pd

from utils.config import PROJECT_ROOT


def csv_to_parquet(filename):
    """Read CSV file and convert to Parquet format.
    
    Args:
        filename: Name of the CSV file (without path or extension)
    """
    # Define paths
    input_file = PROJECT_ROOT / "data" / "raw" / f"{filename}.csv"
    output_file = PROJECT_ROOT / "data" / "raw" / f"{filename}.parquet"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading CSV file from: {input_file}")
    # Read CSV
    df_raw = pd.read_csv(input_file)
    
    print(f"Shape: {df_raw.shape}")
    print(f"Columns: {df_raw.columns.tolist()}")
    
    # Write to Parquet
    print(f"Writing Parquet file to: {output_file}")
    df_raw.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
    
    print(f"Conversion complete! File saved to: {output_file}")
    print(f"Original CSV size: {input_file.stat().st_size / 1024:.2f} KB")
    print(f"Parquet size: {output_file.stat().st_size / 1024:.2f} KB")

    return df_raw

if __name__ == "__main__":
    FILENAME = "infodengue_capitals_subsetBR"
    csv_to_parquet(filename = FILENAME)
