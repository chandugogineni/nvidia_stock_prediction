import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TEST_FILE = "data/processed/test.csv"


df = pd.read_csv(TEST_FILE)


# ============================================================
# BASELINE
# ============================================================

# Assume tomorrow's return is 0%
baseline_return = 0.0


# Predicted price = today's close
baseline_price = df["Close"].values


actual_price = df["Next_Close"].values


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    actual_price,
    baseline_price
)

rmse = (
    mean_squared_error(
        actual_price,
        baseline_price
    ) ** 0.5
)

r2 = r2_score(
    actual_price,
    baseline_price
)


print("=" * 60)
print("NAIVE BASELINE")
print("=" * 60)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")