import pandas as pd


INPUT = (
    "data/processed/nvidia_features_llm.csv"
)


def main():

    df = pd.read_csv(
        INPUT
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    n = len(df)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train = df.iloc[
        :train_end
    ]

    validation = df.iloc[
        train_end:validation_end
    ]

    test = df.iloc[
        validation_end:
    ]

    train.to_csv(
        "data/processed/train_llm.csv",
        index=False
    )

    validation.to_csv(
        "data/processed/validation_llm.csv",
        index=False
    )

    test.to_csv(
        "data/processed/test_llm.csv",
        index=False
    )

    print(
        "LLM datasets created."
    )

    print(
        "Train:",
        train.shape
    )

    print(
        "Validation:",
        validation.shape
    )

    print(
        "Test:",
        test.shape
    )


if __name__ == "__main__":
    main()