import pandas as pd
import os

INPUT_FILE = "data/raw/nvidia_historical_data.csv"
OUTPUT_FILE = "data/processed/nvidia_cleaned.csv"


# Load data
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Convert Date column
# --------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"], utc=True)

# --------------------------------------------------
# 2. Sort chronologically
# --------------------------------------------------

df = df.sort_values("Date")

# --------------------------------------------------
# 3. Remove duplicate dates
# --------------------------------------------------

df = df.drop_duplicates(subset=["Date"])

# --------------------------------------------------
# 4. Check missing values
# --------------------------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# 5. Handle missing values
# --------------------------------------------------

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

df = df.dropna(
    subset=numeric_columns
)

# --------------------------------------------------
# 6. Reset index
# --------------------------------------------------

df = df.reset_index(drop=True)

# --------------------------------------------------
# 7. Save cleaned data
# --------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCleaned shape:", df.shape)

print("\nDate range:")
print(df["Date"].min())
print(df["Date"].max())

print(f"\nSaved cleaned data to: {OUTPUT_FILE}")