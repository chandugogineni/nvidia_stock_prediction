import os
import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


FEATURES = [
    "High_Low_Range",
    "Volume_Change",
    "Momentum_20",
    "Return_Lag_1",
    "MA_20_Ratio",
    "Volatility_20",
    "Volatility_5",
    "Relative_Volume",
    "Return_Lag_10",
    "Momentum_10",
    "Momentum_5",
    "Return_Lag_5",
    "Volatility_50",
    "MA_50_Ratio",
    "Open_Close_Range",
    "Daily_Return",
    "RSI_14",
    "Log_Volume",
    "Return_Lag_2",
    "Return_Lag_3",
    "MA_5_Ratio"
]

TARGET = "Target_Return"


def train():

    train = pd.read_csv(
        "data/processed/train.csv"
    )

    validation = pd.read_csv(
        "data/processed/validation.csv"
    )

    test = pd.read_csv(
        "data/processed/test.csv"
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    print(
        "Training XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation
            )
        ],
        verbose=False
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    actual_direction = (
        y_test > 0
    )

    predicted_direction = (
        predictions > 0
    )

    directional_accuracy = (
        actual_direction ==
        predicted_direction
    ).mean()
    
    print("\nXGBOOST BASELINE")
    print("=" * 60)

    print(
        f"MAE: {mae:.6f}"
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print(
        f"R²: {r2:.6f}"
    )

    print(
        f"Directional Accuracy: "
        f"{directional_accuracy * 100:.2f}%"
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/xgboost.pkl"
    )

    print(
        "\nSaved: models/xgboost.pkl"
    )


if __name__ == "__main__":
    train()