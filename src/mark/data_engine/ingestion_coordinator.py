from datetime import date, timedelta
import pandas as pd
from pathlib import Path
from mark.data_engine.validate_nifty_data import validate_market_data
import os
import tempfile
from mark.data_engine.dataextract import MarketDataProvider, YahooFinanceProvider


def get_fetch_start_date(existing_df :pd.DataFrame|None, initial_start: str, overlap_days:int =7,)->str:


        if existing_df is None or existing_df.empty:
            return initial_start

        latest_data =pd.to_datetime(existing_df["Date"],errors="coerce").max()

        if pd.isna(latest_data):
            raise ValueError("Exisiting data has no valid Date column")
        fetch_start = latest_data - timedelta(days=overlap_days)

        return fetch_start.strftime("%Y-%m-%d")




def load_existing_data(file_path: Path) -> pd.DataFrame | None:

    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)

    return df

def merge_market_data(existing_df: pd.DataFrame,new_df: pd.DataFrame ) -> pd.DataFrame:

    combined_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )

    combined_df["Date"] = pd.to_datetime(
        combined_df["Date"],
        errors="coerce"
    )

    if combined_df["Date"].isna().any():
        raise ValueError("Invalid dates found during merging")

    combined_df = combined_df.drop_duplicates(
        subset=["Date"],
        keep="last" #it removes the duplicates from the first data not new one
    )
    combined_df = combined_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    return combined_df


def save_market_data(df: pd.DataFrame, file_path: Path)->None:
    file_path.parent.mkdir(parents =True, exist_ok =True)
    temp_path = None
    try:

        with tempfile.NamedTemporaryFile(
            mode = "w",
            suffix = ".tmp",
            dir = file_path.parent,
            delete = False,
            encoding="utf-8",
            newline=""

        )as temp_file:


            temp_path = Path(temp_file.name)
            df.to_csv(temp_path,index = False)

        os.replace(temp_path, file_path)

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()




def update_market_data(
    provider: MarketDataProvider,
    symbol: str,
    file_path: Path,
    initial_start: str,
    end: str,
    interval: str = "1d",
    overlap_days: int = 7,
) -> pd.DataFrame:

    existing_data = load_existing_data(file_path)

    fetch_start = get_fetch_start_date(
        existing_df=existing_data,
        initial_start=initial_start,
        overlap_days=overlap_days,
    )

    print(f"Fetching {symbol} from {fetch_start} to {end}")

    new_data = provider.fetch(
        symbol=symbol,
        start=fetch_start,
        end=end,
        interval=interval,
    )

    if existing_data is None:
        merged_data = merge_market_data(
            pd.DataFrame(columns=new_data.columns),
            new_data,
        )
    else:
        merged_data = merge_market_data(
            existing_data,
            new_data,
        )

    validation_report = validate_market_data(merged_data)

    if not validation_report["is_valid"]:
        raise ValueError(
            f"Market data validation failed: "
            f"{validation_report['issues']}"
        )

    print("Market data validation passed.")

    save_market_data(
        df=merged_data,
        file_path=file_path,
    )

    print(f"Market data saved: {file_path}")

    return merged_data


if __name__ == "__main__":

    project_dir = Path.home() / "MARK"

    #  historical data file
    data_file = (
        project_dir
        / "data"
        / "raw"
        / "market"
        / "nifty50_daily_raw.csv"
    )

    #Finance provider
    provider = YahooFinanceProvider()

    end_date = (
        date.today() + timedelta(days=1)
    ).isoformat()

    updated_data = update_market_data(
        provider=provider,
        symbol="^NSEI",
        file_path=data_file,
        initial_start="2011-10-01",
        end=end_date,
        interval="1d",
        overlap_days=7,
    )

    print("\n--- MARK INGESTION REPORT ---")
    print("Total rows:", len(updated_data))
    print("First date:", updated_data["Date"].min())
    print("Latest date:", updated_data["Date"].max())
    print("Data saved to:", data_file)

    
        
