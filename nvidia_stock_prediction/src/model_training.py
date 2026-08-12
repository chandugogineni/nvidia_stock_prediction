import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_FILE = "models/linear_regression.pkl"


# ============================================================
# 1. LOAD DATA
# ============================================================

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))


# ============================================================
# 2. FEATURES
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


# ============================================================
# 3. X / y
# ============================================================

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]


print("\nTraining features:", X_train.shape)
print("Training target:", y_train.shape)

print("Testing features:", X_test.shape)
print("Testing target:", y_test.shape)


# ============================================================
# 4. TARGET INFORMATION
# ============================================================

print("\nTrain target range:")
print(y_train.min(), y_train.max())

print("\nTest target range:")
print(y_test.min(), y_test.max())


# ============================================================
# 5. MODEL PIPELINE
# ============================================================

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# 6. TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 7. PREDICT RETURN
# ============================================================

predicted_returns = model.predict(
    X_test
)


# ============================================================
# 8. CONVERT RETURN TO PRICE
# ============================================================

predicted_prices = (
    test_data["Close"].values
    * (1 + predicted_returns)
)

actual_prices = (
    test_data["Next_Close"].values
)


# ============================================================
# 9. PRICE METRICS
# ============================================================

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

rmse = (
    mean_squared_error(
        actual_prices,
        predicted_prices
    ) ** 0.5
)

r2 = r2_score(
    actual_prices,
    predicted_prices
)


print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 10. RETURN METRICS
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

print("\n")
print("=" * 60)
print("RETURN PREDICTION PERFORMANCE")
print("=" * 60)

print(f"Return MAE  : {return_mae:.6f}")
print(f"Return RMSE : {return_rmse:.6f}")
print(f"Return R²   : {return_r2:.4f}")


# ============================================================
# 11. DIRECTIONAL ACCURACY
# ============================================================

actual_direction = (
    y_test.values > 0
)

predicted_direction = (
    predicted_returns > 0
)

direction_accuracy = (
    actual_direction == predicted_direction
).mean()


print("\n")
print("=" * 60)
print("DIRECTIONAL PERFORMANCE")
print("=" * 60)

print(
    f"Directional Accuracy: "
    f"{direction_accuracy:.2%}"
)


# ============================================================
# 12. SAMPLE PREDICTIONS
# ============================================================

results = test_data[
    [
        "Date",
        "Close",
        "Next_Close",
        "Target_Return"
    ]
].copy()

results["Predicted_Return"] = predicted_returns

results["Predicted_Next_Close"] = predicted_prices


print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    results.tail(20)
    .to_string(index=False)
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)