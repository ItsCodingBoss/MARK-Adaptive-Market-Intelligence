

from pathlib import Path

import pandas as pd


def validate_market_data(df: pd.DataFrame) -> dict:
    """
    Validate the basic structural quality of a market DataFrame.

    Does not modify the original DataFrame.
    """

    if df.empty:
        return {
            "is_valid": False,
            "issues": ["Dataset is empty"],
            "row_count": 0,
        }

    if "Date" not in df.columns:
        return {
            "is_valid": False,
            "issues": ["Date column is missing"],
            "row_count": len(df),
        }

    # Work on a copy to protect the original DataFrame
    working_df = df.copy()

    working_df["Date"] = pd.to_datetime(
        working_df["Date"],
        errors="coerce"
    )

    invalid_dates = int(working_df["Date"].isna().sum())

    duplicate_dates = int(
        working_df["Date"].dropna().duplicated().sum()
    )

    missing_values = {
        column: int(count)
        for column, count in working_df.isna().sum().items()
    }

    valid_dates = (
        working_df["Date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    date_gaps = valid_dates.diff().dt.days.dropna()

    largest_gap = (
        int(date_gaps.max())
        if not date_gaps.empty
        else None
    )

    largest_gaps = date_gaps.nlargest(10)

    gap_report = [
        {
            "previous_date": valid_dates.iloc[index - 1].date().isoformat(),
            "current_date": valid_dates.iloc[index].date().isoformat(),
            "gap_days": int(gap),
        }
        for index, gap in largest_gaps.items()
    ]
    
    ohlc_columns = ["Open", "High", "Low", "Close"]

    ohlc_available = all(
        column in working_df.columns
        for column in ohlc_columns
    )

    invalid_ohlc_rows = 0
    invalid_price_rows = 0

    if ohlc_available:

        # Convert OHLC columns to numeric for checking
        ohlc = working_df[ohlc_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        complete_ohlc = ohlc.notna().all(axis=1)

        invalid_ohlc = (
            (ohlc["High"] < ohlc["Low"])
            | (ohlc["High"] < ohlc["Open"])
            | (ohlc["High"] < ohlc["Close"])
            | (ohlc["Low"] > ohlc["Open"])
            | (ohlc["Low"] > ohlc["Close"])
        )

        invalid_prices = (ohlc <= 0).any(axis=1)

        invalid_ohlc_rows = int(
            (invalid_ohlc & complete_ohlc).sum()
        )

        invalid_price_rows = int(
            (invalid_prices & complete_ohlc).sum()
        )



    issues = []

    if invalid_dates > 0:
        issues.append("Invalid dates detected")

    if duplicate_dates > 0:
        issues.append("Duplicate dates detected")
    if ohlc_available and invalid_ohlc_rows > 0:
        issues.append("Invalid OHLC relationships detected")

    if ohlc_available and invalid_price_rows > 0:
        issues.append("Zero or negative OHLC prices detected")    

    #  Return the structured report
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "row_count": len(working_df),
        "invalid_dates": invalid_dates,
        "duplicate_dates": duplicate_dates,
        "missing_values": missing_values,
        "largest_calendar_gap_days": largest_gap,
        "largest_gaps": gap_report,
        "ohlc_available": ohlc_available,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "invalid_price_rows": invalid_price_rows,
    }


def main():

    project_dir = Path.home() / "MARK"

    data_file = (
        project_dir
        / "data"
        / "processed"
        / "market"
        / "nifty50_daily_basic.csv"
    )

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}"
        )

    df = pd.read_csv(data_file)

    report = validate_market_data(df)

    print("\n--- Market Data Validation Report ---")

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
