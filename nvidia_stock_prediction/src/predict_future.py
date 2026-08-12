import os
import joblib
import pandas as pd


MODEL_FILE = "models/xgboost_llm.pkl"
DATA_FILE = "data/processed/nvidia_features_llm.csv"


# ============================================================
# LLM FEATURES
# ============================================================

LLM_FEATURES = [
    "LLM_Sentiment",
    "LLM_Market_Impact",
    "LLM_AI_Demand",
    "LLM_Regulatory_Risk",
    "LLM_Earnings_Outlook",
    "LLM_Supply_Chain_Risk",
    "LLM_Competition_Risk",
    "LLM_Confidence",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NVIDIA FUTURE PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Data not found: {DATA_FILE}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(MODEL_FILE)

    print("\nModel loaded:")
    print(MODEL_FILE)

    # --------------------------------------------------------
    # IMPORTANT:
    # Get EXACT feature order from trained XGBoost model
    # --------------------------------------------------------

    MODEL_FEATURES = model.feature_names_in_.tolist()

    print("\nModel expects:")
    print(f"{len(MODEL_FEATURES)} features")

    for i, feature in enumerate(MODEL_FEATURES, 1):
        print(f"{i:2}. {feature}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing:

        print("\nERROR: Missing features:")

        for feature in missing:
            print(" -", feature)

        print("\nAvailable columns:")
        print(df.columns.tolist())

        raise ValueError(
            "Required prediction features are missing."
        )

    # --------------------------------------------------------
    # Get latest row
    # --------------------------------------------------------

    latest = df.iloc[-1].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # Use EXACT SAME feature order as training
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[latest[feature] for feature in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    # --------------------------------------------------------
    # Verify feature order
    # --------------------------------------------------------

    print("\nPrediction feature order:")

    for i, feature in enumerate(X.columns, 1):
        print(f"{i:2}. {feature}")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_return = model.predict(X)[0]

    # --------------------------------------------------------
    # Current price
    # --------------------------------------------------------

    current_price = float(
        latest["Close"]
    )

    # --------------------------------------------------------
    # Predicted next price
    # --------------------------------------------------------

    predicted_price = (
        current_price *
        (1 + prediction_return)
    )

    # --------------------------------------------------------
    # Trading signal
    # --------------------------------------------------------

    if prediction_return > 0.01:

        signal = "BUY"

    elif prediction_return < -0.01:

        signal = "SELL"

    else:

        signal = "HOLD"

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Latest market date : "
        f"{latest['Date']}"
    )

    print(
        f"Current price      : "
        f"${current_price:.2f}"
    )

    print(
        f"Predicted return   : "
        f"{prediction_return:.4%}"
    )

    print(
        f"Predicted price    : "
        f"${predicted_price:.2f}"
    )

    print(
        f"Signal             : "
        f"{signal}"
    )

    # --------------------------------------------------------
    # LLM Features
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("LLM FEATURES")
    print("=" * 60)

    for feature in LLM_FEATURES:

        if feature in latest.index:

            value = latest[feature]

            print(
                f"{feature:<28}: "
                f"{float(value):.4f}"
            )

    # --------------------------------------------------------
    # Technical features
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("LATEST MARKET DATA")
    print("=" * 60)

    print(
        f"Open               : "
        f"${latest['Open']:.2f}"
    )

    print(
        f"High               : "
        f"${latest['High']:.2f}"
    )

    print(
        f"Low                : "
        f"${latest['Low']:.2f}"
    )

    print(
        f"Close              : "
        f"${latest['Close']:.2f}"
    )

    print(
        f"Daily Return       : "
        f"{latest['Daily_Return']:.4%}"
    )

    print(
        f"RSI                 : "
        f"{latest['RSI_14']:.2f}"
    )

    print(
        f"Momentum 20         : "
        f"{latest['Momentum_20']:.4%}"
    )

    print(
        f"Volatility 20       : "
        f"{latest['Volatility_20']:.4f}"
    )


if __name__ == "__main__":
    main()