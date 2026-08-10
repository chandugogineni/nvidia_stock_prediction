import pandas as pd
import os


INPUT_FILE = "data/news/nvidia_news.csv"

OUTPUT_FILE = "data/news/nvidia_documents.csv"


def create_documents():

    df = pd.read_csv(INPUT_FILE)

    df["title"] = df["title"].fillna("")

    df["summary"] = df["summary"].fillna("")

    documents = []

    for _, row in df.iterrows():

        text = (
            f"NVIDIA News\n\n"
            f"Title: {row['title']}\n\n"
            f"Summary: {row['summary']}\n\n"
            f"Published: {row['published_at']}"
        )

        documents.append({
            "ticker": row["ticker"],
            "published_at": row["published_at"],
            "title": row["title"],
            "url": row["url"],
            "text": text
        })

    documents_df = pd.DataFrame(
        documents
    )

    documents_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Created {len(documents_df)} documents"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    create_documents()