
import pandas as pd
import json
import os
import time
import ollama


MODEL_NAME = "llama3.2:3b"

INPUT_FILE = "data/processed/nvidia_features.csv"
OUTPUT_FILE = "data/processed/nvidia_llm_features.csv"


def create_prompt(row):

    return f"""
You are an AI financial research assistant analyzing NVIDIA (NVDA).

Analyze the following historical market information.

Date: {row['Date']}

Open: {row['Open']}
High: {row['High']}
Low: {row['Low']}
Close: {row['Close']}
Volume: {row['Volume']}

Daily Return: {row['Daily_Return']}
MA5 Ratio: {row['MA_5_Ratio']}
MA20 Ratio: {row['MA_20_Ratio']}
MA50 Ratio: {row['MA_50_Ratio']}

Volatility 20: {row['Volatility_20']}
High Low Range: {row['High_Low_Range']}
Open Close Range: {row['Open_Close_Range']}
Volume Change: {row['Volume_Change']}
Log Volume: {row['Log_Volume']}

Return ONLY valid JSON.

Use values between -1 and 1 for directional/impact/risk scores.

JSON format:

{{
    "LLM_Sentiment": 0.0,
    "LLM_Market_Impact": 0.0,
    "LLM_AI_Demand": 0.0,
    "LLM_Regulatory_Risk": 0.0,
    "LLM_Earnings_Outlook": 0.0,
    "LLM_Supply_Chain_Risk": 0.0,
    "LLM_Competition_Risk": 0.0,
    "LLM_Confidence": 0.0
}}

Definitions:

LLM_Sentiment:
-1 = very bearish
 0 = neutral
+1 = very bullish

LLM_Market_Impact:
-1 = strongly negative expected market impact
 0 = neutral
+1 = strongly positive expected market impact

LLM_AI_Demand:
-1 = declining AI demand
 0 = neutral
+1 = very strong AI demand

LLM_Regulatory_Risk:
-1 = low regulatory risk
 0 = moderate/neutral
+1 = very high regulatory risk

LLM_Earnings_Outlook:
-1 = poor earnings outlook
 0 = neutral
+1 = very strong earnings outlook

LLM_Supply_Chain_Risk:
-1 = low supply chain risk
 0 = moderate risk
+1 = very high supply chain risk

LLM_Competition_Risk:
-1 = low competition risk
 0 = moderate risk
+1 = very high competition risk

LLM_Confidence:
0 = no confidence
1 = very high confidence

Do not include explanations.
Return JSON only.
"""


def call_ollama(prompt):

    try:

        response = ollama.chat(
            model=MODEL_NAME,
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

        return response["message"]["content"]

    except Exception as e:

        print(f"\nOllama error: {e}")

        return None


def parse_json(response):

    default = {
        "LLM_Sentiment": 0.0,
        "LLM_Market_Impact": 0.0,
        "LLM_AI_Demand": 0.0,
        "LLM_Regulatory_Risk": 0.0,
        "LLM_Earnings_Outlook": 0.0,
        "LLM_Supply_Chain_Risk": 0.0,
        "LLM_Competition_Risk": 0.0,
        "LLM_Confidence": 0.0
    }

    if response is None:
        return default

    try:

        response = response.strip()

        # Remove markdown code fences
        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()

        data = json.loads(response)

        result = {}

        for key in default:

            value = data.get(key, 0.0)

            try:
                value = float(value)
            except:
                value = 0.0

            # Prevent extreme values
            value = max(-1.0, min(1.0, value))

            result[key] = value

        return result

    except Exception as e:

        print("\nJSON parsing error:", e)
        print("Response:", response)

        return default


def main():

    print("=" * 70)
    print("OLLAMA NVIDIA LLM FEATURE GENERATION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)
    df = df.head(20)

    print("Input rows:", len(df))

    results = []

    for index, row in df.iterrows():

        print(
            f"Processing {index + 1}/{len(df)}",
            end="\r"
        )

        prompt = create_prompt(row)

        response = call_ollama(prompt)

        result = parse_json(response)

        results.append(result)

        # Small delay
        time.sleep(0.05)

    print()

    llm_df = pd.DataFrame(results)

    final_df = pd.concat(
        [
            df.reset_index(drop=True),
            llm_df.reset_index(drop=True)
        ],
        axis=1
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("LLM FEATURE GENERATION COMPLETE")
    print("=" * 70)

    print("Output:", OUTPUT_FILE)

    print()
    print("LLM columns:")

    for column in llm_df.columns:
        print("-", column)

    print()
    print("Output shape:", final_df.shape)


if __name__ == "__main__":
    main()
