import json
import re

import ollama


MODEL = "llama3.2:3b"


def call_ollama(
    prompt
):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return (
        response["message"]["content"]
    )


def extract_json(
    text
):

    # Remove markdown fences
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:

        raise ValueError(
            "Could not find JSON in LLM response:\n"
            + text
        )

    return json.loads(
        match.group()
    )


def analyze_news(
    news_context
):

    prompt = f"""
You are a financial NLP feature extraction
system for NVIDIA Corporation (NVDA).

Analyze the following retrieved NVIDIA news.

NEWS:
{news_context}

Return ONLY valid JSON.

Use the following numerical ranges:

sentiment:
-1 = extremely negative
 0 = neutral
+1 = extremely positive

market_impact:
-1 = strongly negative
 0 = neutral
+1 = strongly positive

ai_demand:
-1 = sharply declining AI demand
 0 = neutral
+1 = extremely strong AI demand

regulatory_risk:
-1 = very low regulatory risk
 0 = neutral
+1 = extremely high regulatory risk

earnings_outlook:
-1 = very negative
 0 = neutral
+1 = very positive

supply_chain_risk:
0 = no meaningful risk
1 = extremely high risk

competition_risk:
0 = no meaningful risk
1 = extremely high risk

confidence:
0 = low confidence
1 = very high confidence

Return exactly this JSON structure:

{{
    "sentiment": 0.0,
    "market_impact": 0.0,
    "ai_demand": 0.0,
    "regulatory_risk": 0.0,
    "earnings_outlook": 0.0,
    "supply_chain_risk": 0.0,
    "competition_risk": 0.0,
    "confidence": 0.0
}}
"""

    response = call_ollama(
        prompt
    )

    return extract_json(
        response
    )