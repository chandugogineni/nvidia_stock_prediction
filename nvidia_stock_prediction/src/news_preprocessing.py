import pandas as pd


INPUT = (
    "data/news/nvidia_news.csv"
)

OUTPUT = (
    "data/news/nvidia_documents.csv"
)


def create_documents():

    df = pd.read_csv(
        INPUT
    )

    documents = []

    for _, row in df.iterrows():

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

        published_at = str(
            row.get(
                "published_at",
                ""
            )
        )

        text = f"""
NVIDIA News

Title:
{title}

Published:
{published_at}

Summary:
{summary}
""".strip()

        documents.append(
            {
                "title": title,
                "published_at": published_at,
                "url": row.get(
                    "url"
                ),
                "text": text
            }
        )

    result = pd.DataFrame(
        documents
    )

    result.to_csv(
        OUTPUT,
        index=False
    )

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    create_documents()