import os
import time

import pandas as pd

from rag_retriever import (
    NVIDIARetriever
)

from ollama_llm import (
    analyze_news
)


OUTPUT_FILE = (
    "data/processed/llm_features.csv"
)


def generate():

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    retriever = (
        NVIDIARetriever()
    )

    news = pd.read_csv(
        "data/news/nvidia_news.csv"
    )

    rows = []

    for i, row in news.iterrows():

        title = str(
            row.get(
                "title",
                ""
            )
        )

        summary = str(
            row.get(
                "summary",
                ""
            )
        )

        published_at = row.get(
            "published_at"
        )

        text = (
            f"Title: {title}\n"
            f"Summary: {summary}"
        )

        try:

            results = retriever.search(
                text,
                top_k=5
            )

            context_parts = []

            for result in results:

                context_parts.append(
                    result["text"]
                )

            context = (
                "\n\n".join(
                    context_parts
                )
            )

            features = analyze_news(
                context
            )

            rows.append(
                {
                    "published_at":
                        published_at,

                    "title":
                        title,

                    "LLM_Sentiment":
                        features["sentiment"],

                    "LLM_Market_Impact":
                        features["market_impact"],

                    "LLM_AI_Demand":
                        features["ai_demand"],

                    "LLM_Regulatory_Risk":
                        features["regulatory_risk"],

                    "LLM_Earnings_Outlook":
                        features["earnings_outlook"],

                    "LLM_Supply_Chain_Risk":
                        features["supply_chain_risk"],

                    "LLM_Competition_Risk":
                        features["competition_risk"],

                    "LLM_Confidence":
                        features["confidence"]
                }
            )

            print(
                f"[{i + 1}/{len(news)}] "
                f"Processed: {title[:60]}"
            )

        except Exception as e:

            print(
                f"ERROR: {title}"
            )

            print(e)

        # Prevent excessive local load
        time.sleep(0.1)

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    generate()