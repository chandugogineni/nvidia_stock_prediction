# NVIDIA Stock Prediction (ML + RAG + LLM)

An end-to-end project that predicts next-day NVIDIA (`NVDA`) stock returns by combining classical technical-indicator ML with retrieval-augmented generation over financial news. Started as a straightforward XGBoost regression project and is being extended to test whether LLM-derived signals from news actually add predictive value on top of price/volume features.

> This is a research/education project, not investment advice. Directional accuracy in the low 50s is not a trading edge on its own — see the disclaimer at the bottom.

## Why this project exists

The first version used only price and volume history — moving averages, momentum, volatility, RSI, that kind of thing — and topped out around 53% directional accuracy with XGBoost. Adding more technical indicators didn't move the needle much, which suggests price/volume alone carries a pretty weak signal for next-day returns. So the current phase adds a second information stream: NVIDIA financial news, pulled through a RAG pipeline and summarized into structured features by an LLM (sentiment, perceived AI/GPU demand, regulatory risk, etc.), then fed into the same ML model alongside the technical features.

The core question isn't "can the LLM sound convincing about where NVDA is headed" — it's whether adding that information measurably improves out-of-sample accuracy over the technical-only baseline.

## Architecture

```
NVIDIA market data ──► feature engineering ──► XGBoost ──► ML prediction
                                                                 │
NVIDIA news ──► documents ──► embeddings ──► FAISS ──► RAG retrieval ──► LLM
                                                                 │
                                                          LLM features
                                                                 │
                                              ML prediction + LLM features
                                                                 │
                                                    ensemble model
                                                                 │
                                                       BUY / HOLD / SELL
```

## Objectives

The main objectives are:

Download NVIDIA historical market data.
Clean and preprocess the data.
Generate technical indicators.
Create a next-day prediction target.
Split the data chronologically.
Establish a naive baseline.
Train Linear Regression.
Train Random Forest.
Train XGBoost.
Evaluate regression performance.
Evaluate directional accuracy.
Collect NVIDIA-related news.
Convert news into documents.
Generate embeddings.
Store embeddings in FAISS.
Retrieve relevant news using semantic search.
Use an LLM to analyze retrieved news.
Convert LLM output into structured numerical features.
Combine technical ML features and LLM features.
Train an ensemble model.
Backtest the final strategy.
Compare the final system against a naive baseline and buy-and-hold strategy.

## Tech stack

- **Data**: `yfinance` for OHLCV history and recent news
- **ML**: pandas, numpy, scikit-learn, XGBoost
- **RAG**: OpenAI embeddings (`text-embedding-3-small`), FAISS for vector search
- **LLM**: OpenAI API, used purely for feature extraction — not for generating price predictions directly
- **Config**: python-dotenv

## Features

**Technical (21):** daily return, MA ratios (5/20/50), momentum (5/10/20), volatility (5/20/50), RSI-14, relative volume, return lags (1/2/3/5/10), high-low range, open-close range, volume change, log volume.

**LLM-derived (7):** sentiment, market impact, risk, confidence, AI demand, regulatory risk, earnings outlook — each a numeric score extracted from the most relevant retrieved news for a given date.

## Target variable

```python
Target_Return = (Next_Close - Close) / Close
```

Predicting returns rather than raw price, since price is trivially autocorrelated (that's the 0.998 R² baseline above) and doesn't tell you much on its own.

## Project structure

```
nvidia_stock_prediction/
├── data/
│   ├── raw/nvidia_historical_data.csv
│   ├── processed/{train,validation,test}.csv
│   ├── news/{nvidia_news,nvidia_documents}.csv
│   └── vector_store/{nvidia.faiss, metadata.pkl}
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── ensemble.pkl
└── src/
    ├── download_data.py
    ├── feature_engineering.py
    ├── data_splitting.py
    ├── baseline.py
    ├── model_training.py
    ├── random_forest_training.py
    ├── xgboost_training.py
    ├── news_collection.py
    ├── news_preprocessing.py
    ├── rag_index.py
    ├── rag_retriever.py
    ├── llm_sentiment.py
    ├── llm_features.py
    ├── ensemble_model.py

`requirements.txt

## Current Project Results

The project has already been tested with several models.

Naive Baseline

The naive baseline predicts the next price using the current price.

MAE  : 1.7791
RMSE : 2.9077
R²   : 0.9981

The very high price R² is expected because stock prices are highly autocorrelated.

For this reason, price R² should not be the only evaluation metric.

## Linear Regression

The Linear Regression model was trained using technical features.

Current result:

MAE  : 1.7743
RMSE : 2.9151
R²   : 0.9981

Return MAE  : 0.023691
Return RMSE : 0.032222
Return R²   : -0.0039

Directional Accuracy: 53.92%

Model:

models/linear_regression.pkl
## Random Forest

Random Forest was trained using technical features.

Current result:

Return MAE          : 0.023699
Return RMSE         : 0.032193
Return R²           : -0.0021

Price MAE           : 1.7754
Price RMSE          : 2.9081
Price R²            : 0.9981

Directional Accuracy: 52.91%

Important features included:

High_Low_Range
Volatility_20
MA_50_Ratio
Log_Volume
MA_20_Ratio
Open_Close_Range
Daily_Return
Volume_Change
MA_5_Ratio

Model:

models/random_forest.pkl
## XGBoost

XGBoost is currently the primary machine-learning model.

The project initially used 9 features and was later expanded to 21 features.

The latest feature set includes:

High_Low_Range
Volume_Change
Momentum_20
Return_Lag_1
MA_20_Ratio
Volatility_20
Volatility_5
Relative_Volume
Return_Lag_10
Momentum_10
Momentum_5
Return_Lag_5
Volatility_50
MA_50_Ratio
Open_Close_Range
Daily_Return
RSI_14
Log_Volume
Return_Lag_2
Return_Lag_3
MA_5_Ratio

Latest reported result:

MAE                  : 0.023906
RMSE                 : 0.032865
R²                   : -0.0179
Directional Accuracy : 50.10%

Return MAE           : 0.023039
Return RMSE          : 0.031638
Return R²            : -0.0144
Directional Accuracy : 53.00%

Model:

models/xgboost.pkl

## Important Finding

Adding more technical indicators did not produce a major improvement.

The latest XGBoost model achieved approximately:

53% directional accuracy

This suggests that the available historical price/volume features alone contain only a weak signal for next-day return prediction.

Therefore, the project is being extended with unstructured information, especially financial news.

The goal is not simply to add more technical indicators.

Instead:

Market Information
        +
News Information
        +
LLM Information
        ↓
Improved Feature Representation

## NVIDIA Historical Data

NVIDIA historical data is downloaded using yfinance.

Example:

import yfinance as yf
import pandas as pd

ticker = "NVDA"

nvda = yf.Ticker(ticker)

df = nvda.history(
    period="max"
)

df.reset_index(
    inplace=True
)

print(df.head())
print(df.tail())

print(
    f"Total rows: {len(df)}"
)

df.to_csv(
    "data/raw/nvidia_historical_data.csv",
    index=False
)

print(
    "Saved to data/raw/nvidia_historical_data.csv"
)

The historical dataset contains fields such as:

Date
Open
High
Low
Close
Volume
Dividends
Stock Splits

## Data Preprocessing

The raw NVIDIA data is processed to create machine-learning features.

The processed dataset contains:

Date
Open
High
Low
Close
Volume
Dividends
Stock Splits
Previous_Close
Daily_Return
MA_5
MA_20
MA_50
MA_5_Ratio
MA_20_Ratio
MA_50_Ratio
Volatility_20
High_Low_Range
Open_Close_Range
Volume_Change
Log_Volume
Next_Close
Target_Return
11. Target Variable

The primary prediction target is:

Target_Return

Calculated as:

Target_Return = (Next_Close - Close) / Close

Equivalent:

Target_Return = Next_Close / Close - 1

For example:

Current Close = 200
Next Close    = 210

Return = (210 - 200) / 200

Return = 0.05

Return = +5%
12. Technical Features
Daily Return
df["Daily_Return"] = (
    df["Close"].pct_change()
)
Previous Close
df["Previous_Close"] = (
    df["Close"].shift(1)
)
Moving Average
df["MA_5"] = (
    df["Close"].rolling(5).mean()
)

df["MA_20"] = (
    df["Close"].rolling(20).mean()
)

df["MA_50"] = (
    df["Close"].rolling(50).mean()
)
Moving Average Ratios
df["MA_5_Ratio"] = (
    df["Close"] / df["MA_5"] - 1
)

df["MA_20_Ratio"] = (
    df["Close"] / df["MA_20"] - 1
)

df["MA_50_Ratio"] = (
    df["Close"] / df["MA_50"] - 1
)
High-Low Range
df["High_Low_Range"] = (
    df["High"] - df["Low"]
) / df["Close"]
Open-Close Range
df["Open_Close_Range"] = (
    df["Close"] - df["Open"]
) / df["Open"]
Volume Change
df["Volume_Change"] = (
    df["Volume"].pct_change()
)
Log Volume
import numpy as np

df["Log_Volume"] = (
    np.log1p(df["Volume"])
)

## Momentum Features

Momentum features are calculated using historical returns.

Example:

df["Momentum_5"] = (
    df["Close"].pct_change(5)
)

df["Momentum_10"] = (
    df["Close"].pct_change(10)
)

df["Momentum_20"] = (
    df["Close"].pct_change(20)
)
## Lag Features

Historical returns are included as lagged variables.

df["Return_Lag_1"] = (
    df["Daily_Return"].shift(1)
)

df["Return_Lag_2"] = (
    df["Daily_Return"].shift(2)
)

df["Return_Lag_3"] = (
    df["Daily_Return"].shift(3)
)

df["Return_Lag_5"] = (
    df["Daily_Return"].shift(5)
)

df["Return_Lag_10"] = (
    df["Daily_Return"].shift(10)
)
15. Volatility Features
df["Volatility_5"] = (
    df["Daily_Return"]
    .rolling(5)
    .std()
)

df["Volatility_20"] = (
    df["Daily_Return"]
    .rolling(20)
    .std()
)

df["Volatility_50"] = (
    df["Daily_Return"]
    .rolling(50)
    .std()
)
## RSI

A 14-day RSI can be calculated as:

delta = df["Close"].diff()

gain = delta.clip(
    lower=0
)

loss = -delta.clip(
    upper=0
)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI_14"] = (
    100 - (100 / (1 + rs))
)
## Chronological Data Splitting

Stock-market data should not normally be randomly shuffled.

The project uses chronological splitting.

Example:

Historical Data
       │
       ▼
Training
       │
       ▼
Validation
       │
       ▼
Testing

Current datasets:

data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv

The test set currently extends to:

2026-08-06
## Why Random Train/Test Splitting Is Avoided

Do not use:

train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

for the final time-series experiment.

Random splitting can cause future observations to appear in the training data while older observations appear in the test set.

Instead:

train = df.iloc[:train_end]

validation = df.iloc[
    train_end:validation_end
]

test = df.iloc[
    validation_end:
]

This better represents real-world forecasting.

## RAG Extension

The RAG system adds unstructured NVIDIA financial news.

The RAG pipeline is:

News
 ↓
Documents
 ↓
Embeddings
 ↓
FAISS
 ↓
Semantic Retrieval
 ↓
Relevant News
 ↓
LLM
 ↓
Structured Features
## Embeddings

Each news document is converted into an embedding.

The embedding model is:

text-embedding-3-small

Example:

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

The result is a numerical vector representing the semantic meaning of the article.

## FAISS Vector Database

FAISS is used for semantic similarity search.

The vector database contains:

data/vector_store/
│
├── nvidia.faiss
└── metadata.pkl

The FAISS index stores the embeddings.

The metadata stores:

title
publication date
text
URL

## RAG Retrieval

The retriever receives a query such as:

NVIDIA AI GPU demand and earnings outlook

The query is converted to an embedding.

FAISS searches for the most similar documents.

Example:

results = retrieve_news(
    "NVIDIA AI GPU demand and earnings outlook",
    top_k=5
)

The result is the top five semantically relevant news articles.

## LLM Analysis

The retrieved documents are passed to an LLM.

The LLM is instructed to produce structured information.

Example output:

{
    "sentiment": 0.72,
    "market_impact": 0.68,
    "risk": 0.31,
    "confidence": 0.84,
    "ai_demand": 0.91,
    "regulatory_risk": -0.42,
    "earnings_outlook": 0.77,
    "summary": "Recent NVIDIA news is generally positive."
}

These values become ML features.

## LLM Features

The final LLM feature set is:

LLM_Sentiment
LLM_Market_Impact
LLM_Risk
LLM_Confidence
LLM_AI_Demand
LLM_Regulatory_Risk
LLM_Earnings_Outlook

These features are different from traditional technical indicators.

Technical model:

Price
Volume
Momentum
Volatility
RSI

LLM model:

Sentiment
News impact
AI demand
Regulatory risk
Earnings outlook
## Hybrid Feature Architecture

The final feature matrix becomes:

                    Feature Vector

Technical Features
──────────────────────────────────
Daily_Return
MA_5_Ratio
MA_20_Ratio
MA_50_Ratio
Momentum_5
Momentum_10
Momentum_20
Volatility_5
Volatility_20
Volatility_50
RSI_14
Relative_Volume
Return_Lag_1
Return_Lag_2
Return_Lag_3
Return_Lag_5
Return_Lag_10
High_Low_Range
Open_Close_Range
Volume_Change
Log_Volume

                    +

LLM Features
──────────────────────────────────
LLM_Sentiment
LLM_Market_Impact
LLM_Risk
LLM_Confidence
LLM_AI_Demand
LLM_Regulatory_Risk
LLM_Earnings_Outlook

                    ↓

                XGBoost

                    ↓

             Predicted Return
## Why the LLM Should Not Directly Predict the Price

The system should NOT depend on:

LLM
 ↓
"NVIDIA will rise to $250"

Instead:

News
 ↓
RAG
 ↓
LLM
 ↓
Structured signals
 ↓
Machine Learning Model
 ↓
Prediction

The LLM is used as an information extraction and feature generation system.

The numerical prediction remains the responsibility of the ML model.

## Ensemble Architecture

The final system can use:

XGBoost Prediction
        +
LLM Sentiment
        +
LLM Market Impact
        +
LLM Risk
        +
Technical Features
        ↓
Meta Model
        ↓
Final Prediction

Possible final models:

Logistic Regression
Random Forest
XGBoost
LightGBM
Neural Network

For the first implementation, XGBoost is recommended.

## Final Prediction

The system can generate three possible signals:

BUY
HOLD
SELL

Example:

Predicted Return > +1%
        ↓
       BUY
Predicted Return between -1% and +1%
        ↓
      HOLD
Predicted Return < -1%
        ↓
      SELL

These thresholds should be optimized using validation data rather than selected based on test results.
## The biggest risk: look-ahead bias

If NVIDIA news is published at 6pm on a given trading day, it can't be used to predict that same day's close — only future ones. Every news document needs a timestamp check against the prediction date before it's allowed anywhere near the model. This matters even more for backtesting: the current `yfinance` news feed only returns recent articles, so it's not usable as-is for generating LLM features going back to 1999. A proper historical backtest needs a news source with real publication timestamps, or the RAG features end up being 2026 news attached to 2015 prices — which is a straightforward way to fool yourself into thinking the model works.

## Known limitations / what's not done yet

- News history is recent-only; historical backtesting of the RAG/LLM component isn't valid yet
- No fundamentals (revenue, EPS, margins, guidance) in the feature set yet
- No reranking or metadata filtering on retrieval — it's plain semantic search for now
- Backtest doesn't yet account for transaction costs or slippage
- No experiment tracking or model versioning in place

## Roadmap

- Pull in fundamentals (10-K/10-Q, earnings transcripts, guidance) via RAG
- Add metadata/date filtering + reranking to retrieval instead of raw semantic search
- Source historical news with real timestamps to make backtesting valid
- Add Sharpe ratio, max drawdown, win rate, and transaction costs to the backtest
- Compare naive vs. technical-only vs. technical+LLM vs. full hybrid on the same out-of-sample window

## Disclaimer

This is a research and learning project — not a trading system and not financial advice. ~53% directional accuracy is barely better than a coin flip and says nothing about profitability once you account for transaction costs, slippage, and the fact that markets are shaped by things no model here accounts for (macro conditions, rate decisions, geopolitics, black-swan events). Treat any output from this repo as an experiment, not a signal.

## Current Project Results

The project has already been tested with several models.

Naive Baseline

The naive baseline predicts the next price using the current price.

MAE  : 1.7791
RMSE : 2.9077
R²   : 0.9981

The very high price R² is expected because stock prices are highly autocorrelated.

For this reason, price R² should not be the only evaluation metric.

Linear Regression

The Linear Regression model was trained using technical features.

Current result:

MAE  : 1.7743
RMSE : 2.9151
R²   : 0.9981

Return MAE  : 0.023691
Return RMSE : 0.032222
Return R²   : -0.0039

Directional Accuracy: 53.92%

Model:

models/linear_regression.pkl
Random Forest

Random Forest was trained using technical features.

Current result:

Return MAE          : 0.023699
Return RMSE         : 0.032193
Return R²           : -0.0021

Price MAE           : 1.7754
Price RMSE          : 2.9081
Price R²            : 0.9981

Directional Accuracy: 52.91%

Important features included:

High_Low_Range
Volatility_20
MA_50_Ratio
Log_Volume
MA_20_Ratio
Open_Close_Range
Daily_Return
Volume_Change
MA_5_Ratio

Model:

models/random_forest.pkl
XGBoost

XGBoost is currently the primary machine-learning model.

The project initially used 9 features and was later expanded to 21 features.

The latest feature set includes:

High_Low_Range
Volume_Change
Momentum_20
Return_Lag_1
MA_20_Ratio
Volatility_20
Volatility_5
Relative_Volume
Return_Lag_10
Momentum_10
Momentum_5
Return_Lag_5
Volatility_50
MA_50_Ratio
Open_Close_Range
Daily_Return
RSI_14
Log_Volume
Return_Lag_2
Return_Lag_3
MA_5_Ratio

Latest reported result:

MAE                  : 0.023906
RMSE                 : 0.032865
R²                   : -0.0179
Directional Accuracy : 50.10%

Return MAE           : 0.023039
Return RMSE          : 0.031638
Return R²            : -0.0144
Directional Accuracy : 53.00%


| Model             |         MAE |        RMSE |           R² | Direction Accuracy |
| ----------------- | ----------: | ----------: | -----------: | -----------------: |
| Naive Baseline    |      1.7791 |      2.9077 |       0.9981 |                  — |
| Linear Regression |      1.7743 |      2.9151 |       0.9981 |             53.92% |
| Random Forest     |      1.7754 |      2.9081 |       0.9981 |             52.91% |
| XGBoost           |     0.0237* |     0.0329* |     -0.0182* |             53.10% |
| XGBoost + LLM     | **0.02305** | **0.03163** | **-0.01698** |         **53.39%** |


Model:

models/xgboost.pkl

##Expected End-to-End Flow
                    START
                      │
                      ▼
             Download NVDA Data
                      │
                      ▼
              Data Preprocessing
                      │
                      ▼
             Feature Engineering
                      │
                      ▼
            Train/Validation/Test
                      │
                      ▼
               XGBoost Training
                      │
                      ▼
                ML Prediction
                      │
                      │
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
        NVIDIA News       Historical Data
             │                 │
             ▼                 │
       Text Documents          │
             │                 │
             ▼                 │
         Embeddings            │
             │                 │
             ▼                 │
        FAISS Vector DB        │
             │                 │
             ▼                 │
         RAG Retrieval         │
             │                 │
             ▼                 │
             LLM               │
             │                 │
             ▼                 │
       LLM Features            │
             │                 │
             └────────┬────────┘
                      ▼
               Feature Fusion
                      │
                      ▼
                 Ensemble ML
                      │
                      ▼
              Final Prediction
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
             BUY     HOLD    SELL
                      │
                      ▼
                  Backtest
                      │
                      ▼
                 Evaluation
## Final Goal

The ultimate goal is not simply to achieve a high prediction score.

The goal is to determine whether:

Technical Market Information
          +
Financial News
          +
LLM Information Extraction
          +
RAG Retrieval
          +
Fundamental Information
          ↓
      Machine Learning
          ↓
   improves genuine
out-of-sample prediction

The most important experimental comparison will therefore be:

                    Directional
Model                Accuracy

Naive                  ~50%
Linear Regression      ~54%
Random Forest          ~53%
XGBoost                ~53%
XGBoost + LLM           ?
XGBoost + RAG + LLM     ?

The LLM/RAG system should only be considered successful if it improves strictly out-of-sample performance without introducing look-ahead bias.

## Important Disclaimer

This project is an experimental machine-learning and generative-AI project.

Stock prices are affected by many unpredictable factors, including:

Macroeconomic conditions
Interest rates
Geopolitical events
Market sentiment
Unexpected announcements
Regulatory decisions
Institutional trading
Liquidity
Market structure
Black-swan events

Therefore, no model should be assumed to reliably predict future stock prices.

This project is intended for:

Research
Education
Machine Learning experimentation
RAG experimentation
LLM experimentation
Financial NLP
Time-series modeling

and not as financial advice.# nvidia_stock_prediction
