import pandas as pd

FILE_PATH = "data/raw/nvidia_historical_data.csv"

# Load data
df = pd.read_csv(FILE_PATH)

print("=" * 70)
print("NVIDIA HISTORICAL DATA")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())