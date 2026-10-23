
from pathlib import Path
import pandas as pd


project_dir = Path.home() / "MARK"

input_file = (
    project_dir
    / "data"
    / "processed"
    / "market"
    / "nifty50_daily_basic.csv"
)

output_dir = project_dir / "data" / "processed" / "market"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "nifty50_features.csv"


# Define the features available to the model
feature_columns = [
    "daily_return_pct",
    "return_5d_pct",
    "return_10d_pct",
    "sma_10",
    "sma_20",
    "volatility_10d",
    "volatility_20d",
    "rsi_14",
    "price_vs_sma10_pct",
    "price_vs_sma20_pct",
]


def main():
    # Load the basic market dataset
    df = pd.read_csv(input_file)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Calculate daily returns
    df["daily_return_pct"] = df["Close"].pct_change() * 100

    print(df[["Date", "Close", "daily_return_pct"]].head(10))

    # Calculate 5-observation and 10-observation returns
    df["return_5d_pct"] = df["Close"].pct_change(periods=5) * 100
    df["return_10d_pct"] = df["Close"].pct_change(periods=10) * 100

    print(
        df[
            [
                "Date",
                "Close",
                "daily_return_pct",
                "return_5d_pct",
                "return_10d_pct",
            ]
        ].head(15)
    )

    # Moving averages for trend
    df["sma_10"] = df["Close"].rolling(window=10).mean()
    df["sma_20"] = df["Close"].rolling(window=20).mean()

    print(
        df[["Date", "Close", "sma_10", "sma_20"]].head(25)
    )

    # Calculate rolling volatility using daily returns
    df["volatility_10d"] = (
        df["daily_return_pct"].rolling(window=10).std()
    )

    df["volatility_20d"] = (
        df["daily_return_pct"].rolling(window=20).std()
    )

    print(
        df[
            [
                "Date",
                "daily_return_pct",
                "volatility_10d",
                "volatility_20d",
            ]
        ].tail(15)
    )

    print("\n--- Latest Volatility ---")

    print(
        "10-observation volatility:",
        round(df["volatility_10d"].iloc[-1], 3),
        "%",
    )

    print(
        "20-observation volatility:",
        round(df["volatility_20d"].iloc[-1], 3),
        "%",
    )

    print("\n--- Full Dataset Volatility Summary ---")

    print(
        df[["volatility_10d", "volatility_20d"]]
        .describe()
        .round(3)
    )

    # Calculate 14-observation RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    relative_strength = avg_gain / avg_loss

    df["rsi_14"] = 100 - (100 / (1 + relative_strength))

    print(df[["Date", "Close", "rsi_14"]].tail(15))

    # Measure price distance from moving averages
    df["price_vs_sma10_pct"] = (
        (df["Close"] / df["sma_10"]) - 1
    ) * 100

    df["price_vs_sma20_pct"] = (
        (df["Close"] / df["sma_20"]) - 1
    ) * 100

    print(
        df[
            [
                "Date",
                "Close",
                "sma_10",
                "sma_20",
                "price_vs_sma10_pct",
                "price_vs_sma20_pct",
            ]
        ].tail(10)
    )

    # Target: whether the next observation closes higher
    df["next_close"] = df["Close"].shift(-1)

    df["target_up"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    valid_target = df["next_close"].notna()

    df.loc[valid_target, "target_up"] = (
        df.loc[valid_target, "next_close"]
        > df.loc[valid_target, "Close"]
    ).astype("Int64")

    print("Missing next_close:", int(df["next_close"].isna().sum()))
    print("Missing target_up:", int(df["target_up"].isna().sum()))

    print(
        df[["Date", "Close", "next_close", "target_up"]].tail(3)
    )

    # Separate predictors (X) from the target (y)
    X = df[feature_columns]
    y = df["target_up"]

    print("\n--- Model Dataset Check ---")
    print("Feature count:", len(feature_columns))
    print("Feature names:", feature_columns)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nMissing values in each feature:")
    print(X.isna().sum())

    print("\nMissing target values:", int(y.isna().sum()))

    # Save the complete engineered dataset after all features and target exist
    df.to_csv(output_file, index=False)

    print("\nSaved feature dataset to:", output_file)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())


def prepare_model_data():
    # Load the completed engineered dataset
    model_data = pd.read_csv(output_file)

    model_data["Date"] = pd.to_datetime(model_data["Date"])

    # Keep dates for inspecting the chronological splits
    model_data = model_data[
        ["Date"] + feature_columns + ["target_up"]
    ].copy()

    # Restore the nullable integer target type
    model_data["target_up"] = model_data["target_up"].astype("Int64")

    rows_before = len(model_data)

    # Remove rows with missing features or unknown targets
    model_data = model_data.dropna(
        subset=feature_columns + ["target_up"]
    ).copy()

    rows_after = len(model_data)
    if __name__ == "__main__":
        print("\n--- Model Data Preparation ---")
        print("Rows before:", rows_before)
        print("Rows after:", rows_after)
        print("Rows removed:", rows_before - rows_after)
        print(
            "Remaining missing values:",
            int(model_data.isna().sum().sum()),
        )

    # Chronological train, validation and test split
    n = len(model_data)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_data = model_data.iloc[:train_end].copy()
    val_data = model_data.iloc[train_end:val_end].copy()
    test_data = model_data.iloc[val_end:].copy()

    # Prevent target outcomes from crossing split boundaries
    train_data = train_data.iloc[:-1].copy()
    val_data = val_data.iloc[:-1].copy()

    return train_data, val_data, test_data


if __name__ == "__main__":
    main()

    train_data, val_data, test_data = prepare_model_data()

    print("\n--- Final Dataset Sizes ---")
    print("Training rows:", len(train_data))
    print("Validation rows:", len(val_data))
    print("Testing rows:", len(test_data))

    for name, data in [
        ("Training", train_data),
        ("Validation", val_data),
        ("Testing", test_data),
    ]:
        print(
            f"{name}: "
            f"{data['Date'].min().date()} to "
            f"{data['Date'].max().date()}"
        )

    print("\nTraining target distribution:")
    print(train_data["target_up"].value_counts())

    print("\nTraining target percentages:")
    print(
        train_data["target_up"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )
