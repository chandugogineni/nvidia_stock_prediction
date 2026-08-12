import pandas as pd
import joblib
import os

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TRAIN_FILE = "data/processed/train.csv"
VALIDATION_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_FILE = "models/xgboost.pkl"


# ============================================================
# FEATURES
# ============================================================

features = [
    "Daily_Return",

    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Lag_5",
    "Return_Lag_10",

    "MA_5_Ratio",
    "MA_20_Ratio",
    "MA_50_Ratio",

    "Momentum_5",
    "Momentum_10",
    "Momentum_20",

    "Volatility_5",
    "Volatility_20",
    "Volatility_50",

    "High_Low_Range",
    "Open_Close_Range",

    "Volume_Change",
    "Relative_Volume",
    "Log_Volume",

    "RSI_14"
]
target = "Target_Return"


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)


# ============================================================
# X / Y
# ============================================================

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

X_test = test[features]
y_test = test[target]


print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)


# ============================================================
# MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_validation, y_validation)
    ],
    verbose=False
)


# ============================================================
# VALIDATION
# ============================================================

validation_pred = model.predict(
    X_validation
)

validation_mae = mean_absolute_error(
    y_validation,
    validation_pred
)

validation_rmse = (
    mean_squared_error(
        y_validation,
        validation_pred
    ) ** 0.5
)

validation_r2 = r2_score(
    y_validation,
    validation_pred
)

validation_direction = (
    (y_validation.values > 0)
    ==
    (validation_pred > 0)
).mean()


print("\n" + "=" * 70)
print("VALIDATION PERFORMANCE")
print("=" * 70)

print(
    f"MAE                  : {validation_mae:.6f}"
)

print(
    f"RMSE                 : {validation_rmse:.6f}"
)

print(
    f"R²                   : {validation_r2:.4f}"
)

print(
    f"Directional Accuracy : "
    f"{validation_direction:.2%}"
)


# ============================================================
# FINAL TEST
# ============================================================

test_pred = model.predict(
    X_test
)


test_mae = mean_absolute_error(
    y_test,
    test_pred
)

test_rmse = (
    mean_squared_error(
        y_test,
        test_pred
    ) ** 0.5
)

test_r2 = r2_score(
    y_test,
    test_pred
)

test_direction = (
    (y_test.values > 0)
    ==
    (test_pred > 0)
).mean()


print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)

print(
    f"Return MAE          : {test_mae:.6f}"
)

print(
    f"Return RMSE         : {test_rmse:.6f}"
)

print(
    f"Return R²           : {test_r2:.4f}"
)

print(
    f"Directional Accuracy: "
    f"{test_direction:.2%}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(index=False)
)


# ============================================================
# SAVE
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