import pandas as pd


MARKET_FILE = (
    "data/processed/nvidia_features.csv"
)

LLM_FILE = (
    "data/processed/llm_features.csv"
)

OUTPUT_FILE = (
    "data/processed/nvidia_features_llm.csv"
)


def merge():

    market = pd.read_csv(
        MARKET_FILE
    )

    llm = pd.read_csv(
        LLM_FILE
    )

    market["Date"] = pd.to_datetime(
        market["Date"],
        utc=True
    )

    llm["published_at"] = pd.to_datetime(
        llm["published_at"],
        utc=True,
        errors="coerce"
    )

    llm = (
        llm
        .sort_values(
            "published_at"
        )
    )

    # Use the most recent available
    # LLM information for each market date.
    merged = pd.merge_asof(
        market.sort_values("Date"),
        llm.sort_values("published_at"),
        left_on="Date",
        right_on="published_at",
        direction="backward"
    )

    llm_columns = [
        "LLM_Sentiment",
        "LLM_Market_Impact",
        "LLM_AI_Demand",
        "LLM_Regulatory_Risk",
        "LLM_Earnings_Outlook",
        "LLM_Supply_Chain_Risk",
        "LLM_Competition_Risk",
        "LLM_Confidence"
    ]

    merged[llm_columns] = (
        merged[llm_columns]
        .fillna(0)
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    merge()