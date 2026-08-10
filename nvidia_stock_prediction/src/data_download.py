import yfinance as yf
import pandas as pd

# Download NVIDIA historical data
ticker = "NVDA"
nvda = yf.Ticker(ticker)

# Get max available history (or set a specific period/date range)
df = nvda.history(period="max")  # options: 1d,5d,1mo,6mo,1y,5y,10y,ytd,max

# Alternative: specific date range
# df = nvda.history(start="2015-01-01", end="2026-08-10")

# Reset index so Date becomes a column instead of the index
df.reset_index(inplace=True)

print(df.head())
print(df.tail())
print(f"\nTotal rows: {len(df)}")

# Save to CSV
df.to_csv("nvidia_historical_data.csv", index=False)
print("Saved to nvidia_historical_data.csv")