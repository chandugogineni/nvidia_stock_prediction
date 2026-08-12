import os
import pandas as pd


INPUT_FILE = "data/processed/nvidia_features.csv"

TRAIN_FILE = "data/processed/train.csv"
VALIDATION_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"


def split_data():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. "
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    total = len(df)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train = df.iloc[:train_end].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    print("=" * 60)
    print("DATA SPLITTING")
    print("=" * 60)

    print("Total:", df.shape)

    print("Training:", train.shape)

    print("Validation:", validation.shape)

    print("Testing:", test.shape)

    print("\nDate ranges:")

    print(
        "Train:",
        train["Date"].iloc[0],
        "to",
        train["Date"].iloc[-1]
    )

    print(
        "Validation:",
        validation["Date"].iloc[0],
        "to",
        validation["Date"].iloc[-1]
    )

    print(
        "Test:",
        test["Date"].iloc[0],
        "to",
        test["Date"].iloc[-1]
    )

    print("\nFiles saved:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    split_data()