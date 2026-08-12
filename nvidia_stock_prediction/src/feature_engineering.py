import os
import numpy as np
import pandas as pd


INPUT_FILE = "data/raw/nvidia_historical_data.csv"
OUTPUT_FILE = "data/processed/nvidia_features.csv"


def create_features():

    # --------------------------------------------------
    # Check input
    # --------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Please place nvidia_historical_data.csv inside data/raw/"
        )

    os.makedirs("data/processed", exist_ok=True)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print("Raw data shape:", df.shape)

    # --------------------------------------------------
    # Date
    # --------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True
    )

    df = df.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------
    # Previous close
    # --------------------------------------------------

    df["Previous_Close"] = df["Close"].shift(1)

    # --------------------------------------------------
    # Daily return
    # --------------------------------------------------

    df["Daily_Return"] = df["Close"].pct_change()

    # --------------------------------------------------
    # Moving averages
    # --------------------------------------------------

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # --------------------------------------------------
    # Moving average ratios
    # --------------------------------------------------

    df["MA_5_Ratio"] = df["Close"] / df["MA_5"] - 1
    df["MA_20_Ratio"] = df["Close"] / df["MA_20"] - 1
    df["MA_50_Ratio"] = df["Close"] / df["MA_50"] - 1

    # --------------------------------------------------
    # Volatility
    # --------------------------------------------------

    df["Volatility_5"] = df["Daily_Return"].rolling(5).std()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
    df["Volatility_50"] = df["Daily_Return"].rolling(50).std()

    # --------------------------------------------------
    # Price ranges
    # --------------------------------------------------

    df["High_Low_Range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    df["Open_Close_Range"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    # --------------------------------------------------
    # Volume
    # --------------------------------------------------

    df["Volume_Change"] = df["Volume"].pct_change()

    df["Log_Volume"] = np.log1p(df["Volume"])

    # --------------------------------------------------
    # Relative volume
    # --------------------------------------------------

    volume_ma = df["Volume"].rolling(20).mean()

    df["Relative_Volume"] = (
        df["Volume"] / volume_ma
    )

    # --------------------------------------------------
    # Momentum
    # --------------------------------------------------

    df["Momentum_5"] = df["Close"].pct_change(5)
    df["Momentum_10"] = df["Close"].pct_change(10)
    df["Momentum_20"] = df["Close"].pct_change(20)

    # --------------------------------------------------
    # Return lags
    # --------------------------------------------------

    df["Return_Lag_1"] = df["Daily_Return"].shift(1)
    df["Return_Lag_2"] = df["Daily_Return"].shift(2)
    df["Return_Lag_3"] = df["Daily_Return"].shift(3)
    df["Return_Lag_5"] = df["Daily_Return"].shift(5)
    df["Return_Lag_10"] = df["Daily_Return"].shift(10)

    # --------------------------------------------------
    # RSI
    # --------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI_14"] = (
        100 - (100 / (1 + rs))
    )

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    df["Next_Close"] = df["Close"].shift(-1)

    df["Target_Return"] = (
        df["Next_Close"] / df["Close"] - 1
    )

    # --------------------------------------------------
    # Remove NaN
    # --------------------------------------------------

    df.dropna(inplace=True)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature engineering completed.")

    print("Rows:", len(df))

    print("\nColumns:")

    for column in df.columns:
        print(column)

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    create_features()