import pandas as pd
import os


INPUT_FILE = "data/processed/nvidia_features_llm.csv"

TRAIN_FILE = "data/processed/train_llm.csv"
VALIDATION_FILE = "data/processed/validation_llm.csv"
TEST_FILE = "data/processed/test_llm.csv"


def main():

    print("=" * 60)
    print("LLM DATA SPLITTING")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    # Always sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    print("Total rows:", len(df))

    # 70 / 15 / 15 chronological split

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end]

    validation = df.iloc[
        train_end:validation_end
    ]

    test = df.iloc[
        validation_end:
    ]

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

    print()
    print("Training:", train.shape)
    print("Validation:", validation.shape)
    print("Testing:", test.shape)

    print()

    print("Files saved:")

    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()