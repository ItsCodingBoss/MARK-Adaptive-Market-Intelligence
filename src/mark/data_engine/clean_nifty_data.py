
from pathlib import Path

import pandas as pd

from mark.data_engine.validate_nifty_data import validate_market_data
from mark.data_engine.ingestion_coordinator import save_market_data


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean market data without modifying the original DataFrame.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )

    column_mapping = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj_close": "Adj Close",
        "volume": "Volume"
    }

    df = df.rename(columns=column_mapping)

    required_columns = [
        "Date", "Open", "High", "Low", "Close", "Volume"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    if df["Date"].isna().any():
        raise ValueError("Invalid or missing dates found.")

    numerical_columns = [
        "Open", "High", "Low", "Close", "Adj Close", "Volume"
    ]

    available_columns = [
        col for col in numerical_columns
        if col in df.columns
    ]

    # Convert available numerical columns
    df[available_columns] = df[available_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    for column in required_columns:
        if column != "Date" and df[column].isna().any():
            raise ValueError(
                f"Missing or invalid numerical values in {column}"
            )

    if (df["Volume"] % 1 != 0).any():
        raise ValueError("Fractional volume values found.")

    df["Volume"] = df["Volume"].astype("Int64")

    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates().copy()

    #Sort records chronologically
    df = df.sort_values(
        by="Date",
        ascending=True
    )

    df = df.reset_index(drop=True)

    #  Validate cleaned data
    report = validate_market_data(df)

    if not report["is_valid"]:
        raise ValueError(
            f"Data validation failed: {report['issues']}"
        )

    print("\nData cleaning and validation completed.")

    print(f"Exact duplicate rows removed: {duplicate_count}")
    print(f"Total rows after cleaning: {len(df)}")
    print(f"First date: {df['Date'].min()}")
    print(f"Latest date: {df['Date'].max()}")

    return df


if __name__ == "__main__":

    project_dir = Path.home() / "MARK"

    raw_file = (
        project_dir
        / "data"
        / "raw"
        / "market"
        / "nifty50_daily_raw.csv"
    )

    basic_file = (
        project_dir
        / "data"
        / "processed"
        / "market"
        / "nifty50_daily_basic.csv"
    )

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_file}"
        )

    # Read raw data
    raw_df = pd.read_csv(raw_file)

    print("Raw data loaded.")
    print(f"Original rows: {len(raw_df)}")

    #Clean data
    cleaned_df = clean_market_data(raw_df)

    #Save cleaned data
    save_market_data(cleaned_df, basic_file)

    print("\n--- CLEANING REPORT ---")

    cleaned_df.info()

    print("\nMissing values:")
    print(cleaned_df.isna().sum())

    print("\nFirst five records:")
    print(cleaned_df.head())

    print("\nCleaned data saved to:")
    print(basic_file)
