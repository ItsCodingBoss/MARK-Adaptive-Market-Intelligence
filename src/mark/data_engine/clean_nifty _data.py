import yfinance as yf
from pathlib import Path
import pandas as pd

project = Path.home() / "MARK"

raw_dir = project / "data" / "raw" / "market"
processed_dir = project / "data" / "processed" / "market"

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

raw_file = raw_dir / "nifty50_daily_raw.csv"
basic_file = processed_dir / "nifty50_daily_basic.csv"
final_file = processed_dir / "nifty50_daily_cleaned.csv"
# downloading dataset needed data
df = yf.download(

    "^NSEI",
    start="2011-10-01",
    end="2026-10-10",
    interval="1d",
    auto_adjust=False,
    progress=False

)
if df.empty:
    raise RuntimeError("No download check yahoo finance availability")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# to remove date indexing and remove  un necessary white spaces 
df = df.reset_index()
df.columns = [str(c).strip() for c in df.columns]

df.to_csv(raw_file, index=False)

#it convert the not date to Nat
df["Date"] = pd.to_datetime(df["Date"],errors="coerce")

invalid_dates = df["Date"].isna().sum()
df = df.dropna(subset=["Date"]).copy()

numeric_columns = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume"
]
#change into numeric
for column in numeric_columns:
    if column in df.columns:
        df[column]= pd.to_numeric(df[column],errors="coerce")

exact_duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()   

df = df.sort_values("Date").reset_index(drop=True)
df.to_csv(basic_file, index=False)
df.info()
print(df.describe().round(2))
print(df.isna().sum())

print("Start:", df["Date"].min())
print("End:", df["Date"].max())
print("Rows:", len(df))
invalid_ohlc = (
    (df["High"] < df["Low"])
    | (df["High"] < df[["Open", "Close"]].max(axis=1))
    | (df["Low"] > df[["Open", "Close"]].min(axis=1))
)


print("Invalid OHLC rows:", int(invalid_ohlc.sum()))
print("Invalid dates:", int(invalid_dates))
print("Exact duplicate rows:", int(exact_duplicates))

answer = input(
    "Review additional cleaning options? (yes/no): "
).strip().lower()

if answer not in {"yes", "y"}:
    print("Keeping the basic-cleaned data.")
else:
    # Additional cleaning choices
    pass

