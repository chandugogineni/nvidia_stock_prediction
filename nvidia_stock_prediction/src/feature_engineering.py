import pandas as pd
import numpy as np
import os


INPUT_FILE = "data/processed/nvidia_cleaned.csv"
OUTPUT_FILE = "data/processed/nvidia_features.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# 2. BASIC RETURNS
# ============================================================

df["Previous_Close"] = df["Close"].shift(1)

df["Daily_Return"] = (
    df["Close"] / df["Previous_Close"] - 1
)


# ============================================================
# 3. LAGGED RETURNS
# ============================================================

for lag in [1, 2, 3, 5, 10]:

    df[f"Return_Lag_{lag}"] = (
        df["Daily_Return"].shift(lag)
    )


# ============================================================
# 4. MOVING AVERAGES
# ============================================================

df["MA_5"] = (
    df["Close"]
    .rolling(5)
    .mean()
)

df["MA_20"] = (
    df["Close"]
    .rolling(20)
    .mean()
)

df["MA_50"] = (
    df["Close"]
    .rolling(50)
    .mean()
)


# ============================================================
# 5. NORMALIZED MOVING AVERAGES
# ============================================================

df["MA_5_Ratio"] = (
    df["Close"] / df["MA_5"] - 1
)

df["MA_20_Ratio"] = (
    df["Close"] / df["MA_20"] - 1
)

df["MA_50_Ratio"] = (
    df["Close"] / df["MA_50"] - 1
)


# ============================================================
# 6. MOMENTUM
# ============================================================

df["Momentum_5"] = (
    df["Close"].pct_change(5)
)

df["Momentum_10"] = (
    df["Close"].pct_change(10)
)

df["Momentum_20"] = (
    df["Close"].pct_change(20)
)


# ============================================================
# 7. VOLATILITY
# ============================================================

df["Volatility_5"] = (
    df["Daily_Return"]
    .rolling(5)
    .std()
)

df["Volatility_20"] = (
    df["Daily_Return"]
    .rolling(20)
    .std()
)

df["Volatility_50"] = (
    df["Daily_Return"]
    .rolling(50)
    .std()
)


# ============================================================
# 8. PRICE RANGE
# ============================================================

df["High_Low_Range"] = (
    (df["High"] - df["Low"])
    / df["Close"]
)

df["Open_Close_Range"] = (
    (df["Close"] - df["Open"])
    / df["Open"]
)


# ============================================================
# 9. VOLUME
# ============================================================

df["Volume_Change"] = (
    df["Volume"].pct_change()
)

df["Volume_MA_20"] = (
    df["Volume"]
    .rolling(20)
    .mean()
)

df["Relative_Volume"] = (
    df["Volume"] /
    df["Volume_MA_20"]
)

df["Log_Volume"] = np.log1p(
    df["Volume"]
)


# ============================================================
# 10. RSI
# ============================================================

delta = df["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = (
    gain
    .rolling(14)
    .mean()
)

avg_loss = (
    loss
    .rolling(14)
    .mean()
)

rs = avg_gain / avg_loss

df["RSI_14"] = (
    100 - (100 / (1 + rs))
)


# ============================================================
# 11. TARGET
# ============================================================

df["Next_Close"] = (
    df["Close"].shift(-1)
)

df["Target_Return"] = (
    df["Next_Close"] / df["Close"] - 1
)


# ============================================================
# 12. CLEAN
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = (
    df
    .dropna()
    .reset_index(drop=True)
)


# ============================================================
# 13. TARGET STATISTICS
# ============================================================

print("=" * 70)
print("TARGET STATISTICS")
print("=" * 70)

print(
    df["Target_Return"].describe()
)


# ============================================================
# 14. SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFinal shape:", df.shape)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)