# NVIDIA Stock Prediction with XGBoost, Ollama LLM and RAG

> **Disclaimer:** This is a research/engineering project, not financial advice. Stock prices are highly unpredictable and model accuracy does not guarantee future performance.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Core Data Pipeline](#core-data-pipeline)
- [LLM + Ollama Pipeline](#llm--ollama-pipeline)
- [RAG Architecture](#rag-architecture)
- [Data Splitting](#data-splitting)
- [Model: XGBoost + LLM](#model-xgboost--llm)
- [Results](#results)
- [Baseline Models](#baseline-models)
- [Prediction](#prediction)
- [Data Leakage Prevention](#data-leakage-prevention)
- [Why the LLM Doesn't Automatically Improve Accuracy](#why-the-llm-doesnt-automatically-improve-accuracy)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Status](#project-status)
- [Cleanup / Deprecation Notes](#cleanup--deprecation-notes)

## Overview

This project predicts the next trading-day return and estimated next closing price of NVIDIA (**NVDA**) using a hybrid machine-learning + LLM/RAG architecture.

It combines:

- Historical NVIDIA market data
- Technical / time-series feature engineering
- XGBoost regression, with Random Forest and Linear Regression baselines
- Local LLM inference via **Ollama**
- News collection and preprocessing
- LLM-derived financial signals
- FAISS-based Retrieval-Augmented Generation (RAG)
- Chronological train/validation/test splitting
- Future-date prediction with BUY / HOLD / SELL signal generation

## Architecture

```
NVIDIA Historical Data (Yahoo Finance / CSV)
              │
              ▼
       Data Download
              │
              ▼
     Data Preprocessing
              │
              ▼
    Feature Engineering
              │
   ┌──────────┴──────────┐
   ▼                     ▼
Technical ML Data    News Collection
   │                     │
   │                     ▼
   │              News Preprocessing
   │                     │
   │                     ▼
   │               Ollama Local LLM
   │                     │
   │                     ▼
   │           LLM Features (sentiment,
   │         risk, demand, outlook, ...)
   │                     │
   │                     ▼
   │             FAISS RAG Index
   │                     │
   └──────────┬──────────┘
              ▼
   Merge Technical + LLM Features
              │
              ▼
   Chronological Train/Val/Test Split
              │
              ▼
        XGBoost + LLM
              │
              ▼
         xgboost_llm.pkl
              │
              ▼
        Future Prediction
              │
              ▼
  Predicted Return / Price / Signal
```

## Project Structure

```
nvidia_stock_prediction/
│
├── data/
│   ├── raw/
│   │   └── nvidia_historical_data.csv
│   │
│   └── processed/
│       ├── nvidia_features.csv
│       ├── nvidia_features_llm.csv
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       ├── train_llm.csv
│       ├── validation_llm.csv
│       ├── test_llm.csv
│       ├── predictions.csv
│       └── predictions_llm.csv
│
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost_llm.pkl
│
├── src/
│   ├── pipeline.py
│   ├── data_download.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── data_splitting.py
│   │
│   ├── news_collection.py
│   ├── news_preprocessing.py
│   ├── ollama_llm.py
│   ├── generate_llm_features.py
│   ├── merge_llm_features.py
│   ├── llm_data_splitting.py
│   │
│   ├── rag_index.py
│   ├── rag_retriever.py
│   ├── build_faiss_index.py
│   │
│   ├── baseline.py
│   ├── model_training.py
│   ├── random_forest_training.py
│   ├── xgboost_training.py
│   ├── prediction.py
│   └── predict_future.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### First-time setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> If PowerShell blocks activation with a `running scripts is disabled` error, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry.

### Install and verify Ollama

```powershell
winget install Ollama.Ollama
ollama --version
ollama pull llama3.2
ollama list
ollama run llama3.2
```

## Running the Pipeline

### One-command run (recommended)

Once `pipeline.py` is implemented as the orchestration entry point:

```powershell
python src/pipeline.py
```

Recommended internal sequence (using `subprocess.run(..., check=True)` so the pipeline stops on the first failed step):

1. Download market data
2. Preprocess market data
3. Engineer technical features
4. Collect financial news
5. Preprocess news
6. Build/update FAISS index
7. Generate LLM features with Ollama
8. Merge technical + LLM features
9. Split train/validation/test
10. Train XGBoost + LLM
11. Evaluate model
12. Generate historical predictions
13. Generate latest future prediction

### Manual, step-by-step execution (for debugging)

```powershell
python src/data_download.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/news_collection.py
python src/news_preprocessing.py
python src/build_faiss_index.py
python src/generate_llm_features.py
python src/merge_llm_features.py
python src/llm_data_splitting.py
python src/xgboost_training.py
python src/prediction.py
python src/predict_future.py
```

## Core Data Pipeline

**`data_download.py`** — downloads/updates NVIDIA historical market data.
- Input: Yahoo Finance / external market source
- Output: `data/raw/nvidia_historical_data.csv`

**`data_preprocessing.py`** — cleans and normalizes the raw market dataset.
- Parses dates, sorts chronologically, removes invalid rows, normalizes column names, handles missing values

**`feature_engineering.py`** — builds technical features:

| Category | Features |
|---|---|
| Returns/Momentum | `Daily_Return`, `Momentum_5`, `Momentum_10`, `Momentum_20` |
| Moving averages | `MA_5_Ratio`, `MA_20_Ratio`, `MA_50_Ratio` |
| Volatility | `Volatility_5`, `Volatility_20`, `Volatility_50` |
| Range | `High_Low_Range`, `Open_Close_Range` |
| Volume | `Volume_Change`, `Log_Volume`, `Relative_Volume` |
| Lagged returns | `Return_Lag_1`, `Return_Lag_2`, `Return_Lag_3`, `Return_Lag_5`, `Return_Lag_10` |
| Oscillator | `RSI_14` |

Target: `Target_Return = Next_Close / Close - 1`

Output: `data/processed/nvidia_features.csv`

## LLM + Ollama Pipeline

The LLM does **not** replace XGBoost — it generates numerical features that are combined with technical features:

```
Financial News → Ollama → Structured Financial Signals → XGBoost → Prediction
```

### Current LLM features (used by the trained model)

```
LLM_Sentiment
LLM_Market_Impact
LLM_AI_Demand
LLM_Regulatory_Risk
LLM_Earnings_Outlook
LLM_Supply_Chain_Risk
LLM_Competition_Risk
LLM_Confidence
```

> Do not mix these with unrelated columns such as `market_sentiment`, `market_strength`, `risk_score`, `trend_score`, `reasoning_score` unless the training pipeline is explicitly changed to use them.

## RAG Architecture

```
News → Preprocessing → Chunks/Documents → Embeddings → FAISS Index
     → Retriever → Relevant News → Ollama → Structured Financial Signals
```

Relevant files: `news_collection.py`, `news_preprocessing.py`, `rag_index.py`, `rag_retriever.py`, `build_faiss_index.py`, `ollama_llm.py`

## Data Splitting

This is a time-series problem — **no random train/test splitting**. Data is split chronologically: oldest → training → validation → testing → newest.

Current approximate split (changes as new market data is downloaded):

| Split | Rows |
|---|---|
| Training | 4,815 |
| Validation | 1,032 |
| Testing | 1,032 |

## Model: XGBoost + LLM

Final trained model: `models/xgboost_llm.pkl`

The model expects **29 features** — 21 technical + 8 LLM:

**Technical (21):** `High_Low_Range`, `Volume_Change`, `Momentum_20`, `Return_Lag_1`, `MA_20_Ratio`, `Volatility_20`, `Volatility_5`, `Relative_Volume`, `Return_Lag_10`, `Momentum_10`, `Momentum_5`, `Return_Lag_5`, `Volatility_50`, `MA_50_Ratio`, `Open_Close_Range`, `Daily_Return`, `RSI_14`, `Log_Volume`, `Return_Lag_2`, `Return_Lag_3`, `MA_5_Ratio`

**LLM (8):** `LLM_Sentiment`, `LLM_Market_Impact`, `LLM_AI_Demand`, `LLM_Regulatory_Risk`, `LLM_Earnings_Outlook`, `LLM_Supply_Chain_Risk`, `LLM_Competition_Risk`, `LLM_Confidence`

> The prediction dataframe must use exactly the same feature names and order as training.

## Results

### XGBoost + LLM

| Metric | Value |
|---|---|
| MAE | 0.023050 |
| RMSE | 0.031630 |
| R² | -0.016982 |
| Directional Accuracy | 53.39% |

### XGBoost (technical-only)

| Metric | Value |
|---|---|
| MAE | 0.023906 |
| RMSE | 0.032865 |
| R² | -0.0179 |
| Directional Accuracy | 50.10% |

The LLM experiment is a modest improvement over technical-only XGBoost. It should be described as an **experimental feature-augmentation approach**, not a guarantee of higher accuracy.

## Baseline Models

**Naive baseline** (`baseline.py`)

| Metric | Value |
|---|---|
| MAE | ~1.7791 |
| RMSE | ~2.9077 |
| R² | ~0.9981 |

The very high price R² mostly reflects the high autocorrelation of a stock-price series — for model comparison, return prediction and directional accuracy are more informative.

**Linear Regression** (`model_training.py`) — simple ML baseline for comparing return and price prediction.

**Random Forest** (`random_forest_training.py`)

| Metric | Value |
|---|---|
| Return MAE | 0.023699 |
| Return RMSE | 0.032193 |
| Return R² | -0.0021 |
| Directional Accuracy | 52.91% |

**XGBoost** (`xgboost_training.py`) — two variants: technical-only, and technical + LLM features. The project uses **XGBoost + LLM** as the primary experimental model.

## Prediction

### Historical test predictions — `prediction.py`

Evaluates the model against known historical test data. Output columns: `Date`, `Close`, `Target_Return`, `Predicted_Return`, `Predicted_Next_Close`, `Signal`.

### Future prediction — `predict_future.py`

Loads `models/xgboost_llm.pkl` and uses the latest available market row to produce current price, predicted return, predicted price, and signal.

Signal thresholds (configurable research parameters, not investment recommendations):

```
predicted_return >  1%  → BUY
predicted_return < -1%  → SELL
otherwise               → HOLD
```

### Example future prediction output

```
============================================================
NVIDIA FUTURE PREDICTION
============================================================
Latest market date : 2026-08-10
Current price      : $217.55

Predicted return    : 0.0770%
Predicted price     : $217.72

Signal              : HOLD
```

> The model output is a prediction, not a guaranteed future price.

### Note on future dates

The model cannot directly predict a date like 2026-08-12 or 2026-08-13 unless the required input features for that date exist. If the latest downloaded market data ends on 2026-08-10, later dates are future trading days relative to the dataset — a new prediction requires fresh market data and fresh news run back through the pipeline (technical features + RAG/Ollama features → XGBoost → next-day prediction).

## Data Leakage Prevention

- Sort all data chronologically.
- Create lagged features only from past data.
- Create `Target_Return` from the *next* trading day's close; never use `Next_Close` as an input feature.
- Split chronologically — no shuffling.
- News used for a prediction must have been published before the prediction timestamp; never generate LLM features from future news.
- Do not fit preprocessing/scalers on the full dataset before splitting.
- Keep the test set untouched until final evaluation.

## Why the LLM Doesn't Automatically Improve Accuracy

The LLM is useful for information technical indicators can't directly capture: earnings expectations, AI infrastructure/GPU demand, regulatory developments, supply-chain disruptions, competitive announcements, major customer news, and sentiment around NVIDIA.

However, **LLM ≠ guaranteed prediction accuracy** — LLM features are noisy and may add no predictive value. The correct experiment is a controlled comparison on the *same* unseen test period:

- **Model A:** Technical features → XGBoost
- **Model B:** Technical features + LLM features → XGBoost

## Evaluation Metrics

Don't rely only on price R². Use:

- Return MAE, RMSE, R²
- Directional Accuracy
- Precision, Recall, F1
- Maximum Drawdown
- Sharpe Ratio

For a trading-oriented project, directional accuracy and backtested risk-adjusted performance matter most.

## Project Status

- [x] Historical NVIDIA data
- [x] Technical feature engineering
- [x] Chronological train/validation/test split
- [x] Naive baseline
- [x] Linear Regression
- [x] Random Forest
- [x] XGBoost
- [x] Ollama integration
- [x] LLM financial features
- [x] LLM + XGBoost
- [x] Future prediction script
- [x] RAG architecture
- [x] FAISS architecture
- [ ] `src/pipeline.py` as the single orchestration entry point *(main remaining task)*

## Cleanup / Deprecation Notes

Don't delete files immediately — move old/duplicate experiments into an `archive/` directory first.

- **`split_llm_data.py`** — if `llm_data_splitting.py` already performs the final LLM split correctly, keep only one splitting implementation.
- **`generate_llm_features.py` / `llm_features.py` / `llm_sentiment.py`** — consolidate if they duplicate functionality; `generate_llm_features.py` should be the single entry point for LLM feature generation.
- **`build_faiss_index.py` vs `rag_index.py`** — use `build_faiss_index.py` as the executable pipeline step, and keep `rag_index.py` as the reusable RAG implementation module.
- **`data_splitting.py` vs `llm_data_splitting.py`** — keep both only if they have distinct responsibilities (technical-only vs. technical+LLM datasets); otherwise consolidate.

**Files to keep:** `baseline.py`, `data_download.py`, `data_preprocessing.py`, `feature_engineering.py`, `news_collection.py`, `news_preprocessing.py`, `ollama_llm.py`, `rag_index.py`, `rag_retriever.py`, `build_faiss_index.py`, `generate_llm_features.py`, `merge_llm_features.py`, `llm_data_splitting.py`, `model_training.py`, `random_forest_training.py`, `xgboost_training.py`, `prediction.py`, `predict_future.py`, `pipeline.py` — these are the main project workflow and evaluation components.

## Project Goal

Build a reproducible hybrid financial prediction system:

```
Technical Analysis + Financial News + RAG + Local Ollama LLM + XGBoost
              ↓
    Next-Day NVIDIA Return Prediction
```

The LLM acts as an information-extraction and contextual-signal layer, while XGBoost remains the primary numerical prediction model.
