import pandas as pd
from pathlib import Path

project_dir =Path.home()/"MARK"

data_file =  (
    project_dir
    / "data"
    / "processed"
    / "market"
    / "nifty50_daily_basic.csv"
)

if not  data_file.exists():
    raise FileNotFoundError(f"Dataset not found: {data_file}")

df = pd.read_csv(data_file)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())
# Check missing values in each column
missing_values = df.isna().sum()

print("\n--- Missing Values Report ---")
print(missing_values)

print("\nTotal missing cells:", int(missing_values.sum()))

# Check duplicate dates
if "Date" not in df.columns:
    raise ValueError("The dataset has no Date column.")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

invalid_dates = int(df["Date"].isna().sum())
duplicate_dates = int(df["Date"].duplicated().sum())

print("\n--- Date Quality Report ---")
print("Invalid dates:", invalid_dates)
print("Duplicate dates:", duplicate_dates)



#finding the date gaps in the dataset
date_gaps = df["Date"].diff().dt.days
largest_gaps = date_gaps.nlargest(10)
print("\n--- Trading Date Gap Report ---")
print("Largest calendar-day gap:", int(date_gaps.max()))

# Find the ten largest gaps once
largest_gaps = date_gaps.nlargest(10)

# Create a report with both dates
gap_report = pd.DataFrame({
    "Previous_Date": df.loc[largest_gaps.index - 1, "Date"].values,
    "Current_Date": df.loc[largest_gaps.index, "Date"].values,
    "Gap_Days": largest_gaps.values
})

print("\nLargest gaps:")
print(gap_report)
