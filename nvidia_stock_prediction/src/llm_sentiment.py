import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from rag_retriever import retrieve_news


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


MODEL = "gpt-5.6"


def analyze_news(query):

    documents = retrieve_news(
        query,
        top_k=5
    )

    context = "\n\n".join(
        [
            (
                f"ARTICLE {i + 1}\n"
                f"Title: {doc['title']}\n"
                f"Published: {doc['published_at']}\n"
                f"Content: {doc['text']}"
            )
            for i, doc in enumerate(documents)
        ]
    )

    prompt = f"""
You are a financial news analysis system.

Analyze the retrieved NVIDIA news.

Do NOT invent information.

Based only on the supplied news, produce a structured
assessment of the potential short-term impact on NVIDIA.

Retrieved news:

{context}

Return JSON with exactly these fields:

{{
    "sentiment": -1 to 1,
    "market_impact": -1 to 1,
    "risk": 0 to 1,
    "confidence": 0 to 1,
    "ai_demand": -1 to 1,
    "regulatory_risk": -1 to 1,
    "earnings_outlook": -1 to 1,
    "summary": "short explanation"
}}
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    text = response.output_text

    try:

        result = json.loads(
            text
        )

    except json.JSONDecodeError:

        print(
            "LLM returned non-JSON:"
        )

        print(text)

        raise

    return result


if __name__ == "__main__":

    result = analyze_news(
        "NVIDIA latest earnings, AI demand, GPUs and regulatory risks"
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )