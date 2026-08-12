import os
import json
import pandas as pd
import yfinance as yf


def collect_news():

    os.makedirs(
        "data/news",
        exist_ok=True
    )

    ticker = yf.Ticker(
        "NVDA"
    )

    print(
        "Collecting NVIDIA news..."
    )

    news_items = ticker.news

    rows = []

    for item in news_items:

        content = item.get(
            "content",
            {}
        )

        title = content.get(
            "title"
        )

        summary = content.get(
            "summary"
        )

        pub_date = content.get(
            "pubDate"
        )

        canonical_url = None

        canonical = content.get(
            "canonicalUrl"
        )

        if isinstance(
            canonical,
            dict
        ):
            canonical_url = canonical.get(
                "url"
            )

        rows.append(
            {
                "title": title,
                "summary": summary,
                "published_at": pub_date,
                "url": canonical_url,
                "raw_json": json.dumps(
                    item
                )
            }
        )

    df = pd.DataFrame(
        rows
    )

    output = (
        "data/news/nvidia_news.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        f"News articles: {len(df)}"
    )

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    collect_news()