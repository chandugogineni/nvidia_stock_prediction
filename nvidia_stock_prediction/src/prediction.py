import joblib
import pandas as pd


TECHNICAL_FEATURES = [
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

LLM_FEATURES = [
    "LLM_Sentiment",
    "LLM_Market_Impact",
    "LLM_AI_Demand",
    "LLM_Regulatory_Risk",
    "LLM_Earnings_Outlook",
    "LLM_Supply_Chain_Risk",
    "LLM_Competition_Risk",
    "LLM_Confidence"
]

FEATURES = (
    TECHNICAL_FEATURES
    +
    LLM_FEATURES
)


def predict():

    model = joblib.load(
        "models/xgboost_llm.pkl"
    )

    df = pd.read_csv(
        "data/processed/test_llm.csv"
    )

    X = df[FEATURES]

    predictions = (
        model.predict(X)
    )

    result = df[
        [
            "Date",
            "Close",
            "Target_Return"
        ]
    ].copy()

    result[
        "Predicted_Return"
    ] = predictions

    result[
        "Predicted_Next_Close"
    ] = (
        result["Close"]
        *
        (1 + predictions)
    )

    result[
        "Signal"
    ] = "HOLD"

    result.loc[
        result["Predicted_Return"] > 0.01,
        "Signal"
    ] = "BUY"

    result.loc[
        result["Predicted_Return"] < -0.01,
        "Signal"
    ] = "SELL"

    print(
        result.tail(30).to_string(
            index=False
        )
    )

    result.to_csv(
        "data/processed/predictions_llm.csv",
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "data/processed/predictions_llm.csv"
    )


if __name__ == "__main__":
    predict()