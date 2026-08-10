import yfinance as yf
import pandas as pd
import os


TICKER = "NVDA"

OUTPUT_FILE = "data/news/nvidia_news.csv"


def collect_news():

    ticker = yf.Ticker(TICKER)

    news = ticker.news

    records = []

    for item in news:

        content = item.get("content", {})

        title = content.get("title")

        summary = content.get("summary")

        pub_date = content.get("pubDate")

        url = None

        canonical = content.get("canonicalUrl")

        if canonical:
            url = canonical.get("url")

        records.append({
            "ticker": TICKER,
            "title": title,
            "summary": summary,
            "published_at": pub_date,
            "url": url
        })

    df = pd.DataFrame(records)

    os.makedirs(
        "data/news",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Collected {len(df)} news articles"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    collect_news()