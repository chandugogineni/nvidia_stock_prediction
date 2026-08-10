import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_FILE = "models/random_forest.pkl"


# ============================================================
# LOAD DATA
# ============================================================

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)


# ============================================================
# FEATURES
# ============================================================

features = [
    "Daily_Return",
    "MA_5_Ratio",
    "MA_20_Ratio",
    "MA_50_Ratio",
    "Volatility_20",
    "High_Low_Range",
    "Open_Close_Range",
    "Volume_Change",
    "Log_Volume"
]

target = "Target_Return"


X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("Training Random Forest...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICT
# ============================================================

predicted_returns = model.predict(
    X_test
)


# ============================================================
# PRICE
# ============================================================

predicted_prices = (
    test_data["Close"].values
    * (1 + predicted_returns)
)

actual_prices = (
    test_data["Next_Close"].values
)


# ============================================================
# RETURN METRICS
# ============================================================

return_mae = mean_absolute_error(
    y_test,
    predicted_returns
)

return_rmse = (
    mean_squared_error(
        y_test,
        predicted_returns
    ) ** 0.5
)

return_r2 = r2_score(
    y_test,
    predicted_returns
)


# ============================================================
# PRICE METRICS
# ============================================================

price_mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

price_rmse = (
    mean_squared_error(
        actual_prices,
        predicted_prices
    ) ** 0.5
)

price_r2 = r2_score(
    actual_prices,
    predicted_prices
)


# ============================================================
# DIRECTION
# ============================================================

actual_direction = y_test > 0

predicted_direction = predicted_returns > 0

direction_accuracy = (
    actual_direction.values
    ==
    predicted_direction
).mean()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST PERFORMANCE")
print("=" * 60)

print(f"Return MAE          : {return_mae:.6f}")
print(f"Return RMSE         : {return_rmse:.6f}")
print(f"Return R²           : {return_r2:.4f}")

print()

print(f"Price MAE           : {price_mae:.4f}")
print(f"Price RMSE          : {price_rmse:.4f}")
print(f"Price R²            : {price_r2:.4f}")

print()

print(
    f"Directional Accuracy: "
    f"{direction_accuracy:.2%}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"\nModel saved to: {MODEL_FILE}"
)