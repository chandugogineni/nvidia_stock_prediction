import os
import yfinance as yf
import pandas as pd


def download_nvda_data():

    os.makedirs("data/raw", exist_ok=True)

    ticker = "NVDA"

    print("=" * 60)
    print("Downloading NVIDIA historical data")
    print("=" * 60)

    nvda = yf.Ticker(ticker)

    df = nvda.history(
        period="max",
        auto_adjust=False
    )

    df.reset_index(inplace=True)

    # Keep required columns
    columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits"
    ]

    df = df[columns]

    output_path = (
        "data/raw/nvidia_historical_data.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Rows: {len(df)}")
    print(f"Start: {df['Date'].iloc[0]}")
    print(f"End:   {df['Date'].iloc[-1]}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    download_nvda_data()