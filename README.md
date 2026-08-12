# NVIDIA Stock Prediction with XGBoost, Ollama LLM and RAG

1. Project Overview

This project predicts the next trading-day return and estimated next closing price of NVIDIA (NVDA) using a hybrid machine-learning and LLM/RAG architecture.

The project combines:

Historical NVIDIA market data

Technical/time-series feature engineering

XGBoost regression

Random Forest and Linear Regression baselines

Ollama local LLM inference

News collection and preprocessing

LLM-derived financial signals

FAISS-based Retrieval-Augmented Generation (RAG)

Chronological train/validation/test splitting

Future-date prediction

BUY / HOLD / SELL signal generation

Important: This is a research/engineering project, not financial advice. Stock prices are highly unpredictable and model accuracy does not guarantee future performance.

2. High-Level Architecture

                         ┌─────────────────────────┐
                         │ NVIDIA Historical Data  │
                         │   Yahoo Finance / CSV   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Data Download         │
                         │   data_download.py      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Data Preprocessing      │
                         │ data_preprocessing.py   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Feature Engineering     │
                         │ feature_engineering.py  │
                         └────────────┬────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
                     ▼                                 ▼
          ┌─────────────────────┐          ┌──────────────────────┐
          │ Technical ML Data   │          │ News Collection      │
          │                     │          │ news_collection.py   │
          └──────────┬──────────┘          └──────────┬───────────┘
                     │                                 │
                     │                                 ▼
                     │                    ┌──────────────────────┐
                     │                    │ News Preprocessing   │
                     │                    │ news_preprocessing.py│
                     │                    └──────────┬───────────┘
                     │                               │
                     │                               ▼
                     │                    ┌──────────────────────┐
                     │                    │ Ollama Local LLM     │
                     │                    │ ollama_llm.py        │
                     │                    └──────────┬───────────┘
                     │                               │
                     │                               ▼
                     │                    ┌──────────────────────┐
                     │                    │ LLM Features         │
                     │                    │ sentiment / risk /   │
                     │                    │ demand / outlook     │
                     │                    └──────────┬───────────┘
                     │                               │
                     │                               ▼
                     │                    ┌──────────────────────┐
                     │                    │ FAISS RAG Index      │
                     │                    │ rag_index.py         │
                     │                    │ rag_retriever.py    │
                     │                    └──────────┬───────────┘
                     │                               │
                     └──────────────┬────────────────┘
                                    ▼
                         ┌─────────────────────────┐
                         │ Merge Technical + LLM   │
                         │ Features                │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Chronological Splitting │
                         │ Train / Validation/Test │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ XGBoost + LLM           │
                         │ xgboost_training.py     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ xgboost_llm.pkl         │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Future Prediction       │
                         │ predict_future.py       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Predicted Return        │
                         │ Predicted Price         │
                         │ BUY / HOLD / SELL       │
                         └─────────────────────────┘

3. Final Recommended Project Structure

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

4. Files to Keep

Core data pipeline

data_download.py

Downloads/updates NVIDIA historical market data.

Input:

Yahoo Finance / external market source

Output:

data/raw/nvidia_historical_data.csv

data_preprocessing.py

Cleans and normalizes the raw market dataset.

Responsibilities:

Parse dates

Sort chronologically

Remove invalid rows

Normalize column names

Handle missing values

feature_engineering.py

Creates technical features.

Current features include:

Daily_Return

MA_5_Ratio

MA_20_Ratio

MA_50_Ratio

Volatility_5

Volatility_20

Volatility_50

High_Low_Range

Open_Close_Range

Volume_Change

Log_Volume

Relative_Volume

Momentum_5

Momentum_10

Momentum_20

Return_Lag_1

Return_Lag_2

Return_Lag_3

Return_Lag_5

Return_Lag_10

RSI_14

Target:

Target_Return =
    Next_Close / Close - 1

Output:

data/processed/nvidia_features.csv

5. LLM + Ollama Pipeline

The LLM is not used as a direct replacement for XGBoost.

Instead:

Financial News
      ↓
Ollama
      ↓
Structured Financial Signals
      ↓
XGBoost
      ↓
Prediction

The LLM generates numerical features that are combined with technical features.

LLM features

The current model uses:

LLM_Sentiment
LLM_Market_Impact
LLM_AI_Demand
LLM_Regulatory_Risk
LLM_Earnings_Outlook
LLM_Supply_Chain_Risk
LLM_Competition_Risk
LLM_Confidence

These are the actual LLM features expected by the trained XGBoost model.

Do not mix them with unrelated columns such as:

market_sentiment
market_strength
risk_score
trend_score
reasoning_score

unless the training pipeline is explicitly changed to use those columns.

6. Ollama

Ollama runs the LLM locally.

Example installation:

winget install Ollama.Ollama

Verify:

ollama --version

Download a model:

ollama pull llama3.2

Check installed models:

ollama list

Test:

ollama run llama3.2

The project can then use Ollama through the local API.

7. RAG Architecture

The RAG component provides relevant financial news/context to the LLM.

News
 ↓
Preprocessing
 ↓
Chunks/Documents
 ↓
Embeddings
 ↓
FAISS Index
 ↓
Retriever
 ↓
Relevant News
 ↓
Ollama
 ↓
Structured Financial Signals

Relevant files:

news_collection.py
news_preprocessing.py
rag_index.py
rag_retriever.py
build_faiss_index.py
ollama_llm.py

8. Chronological Data Splitting

This is a time-series problem, so random train/test splitting should not be used.

The project uses:

Older data
    ↓
Training
    ↓
Validation
    ↓
Testing
    ↓
Newest data

Current approximate split:

Training   : 4815 rows
Validation : 1032 rows
Testing    : 1032 rows

The exact row counts change when new market data is downloaded.

9. XGBoost + LLM

The final trained model is:

models/xgboost_llm.pkl

The model expects 29 features:

Technical features

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

LLM features

LLM_Sentiment
LLM_Market_Impact
LLM_AI_Demand
LLM_Regulatory_Risk
LLM_Earnings_Outlook
LLM_Supply_Chain_Risk
LLM_Competition_Risk
LLM_Confidence

Total:

21 technical + 8 LLM = 29 features

The prediction dataframe must use exactly the same feature names and order.

10. Model Results

The current XGBoost + LLM experiment produced approximately:

MAE                  : 0.023050
RMSE                 : 0.031630
R²                   : -0.016982
Directional Accuracy : 53.39%

This is only a modest improvement over the technical-only XGBoost experiment.

Earlier technical-only XGBoost results were approximately:

MAE                  : 0.023906
RMSE                 : 0.032865
R²                   : -0.0179
Directional Accuracy : 50.10%

Therefore the LLM experiment should be described as an experimental feature-augmentation approach rather than claiming that the LLM guarantees higher accuracy.

11. Baseline Models

The project contains several models for comparison.

Naive baseline

baseline.py

Used to establish a simple reference point.

Observed result:

MAE  : ~1.7791
RMSE : ~2.9077
R²   : ~0.9981

The very high price R² is largely a consequence of predicting a highly autocorrelated stock-price series. For model comparison, return prediction and directional accuracy are more informative.

Linear Regression

model_training.py

Purpose:

Establish a simple ML baseline

Compare return prediction

Compare price prediction

Random Forest

random_forest_training.py

Observed experiment:

Return MAE          : 0.023699
Return RMSE         : 0.032193
Return R²           : -0.0021
Directional Accuracy: 52.91%

XGBoost

xgboost_training.py

Two variants were experimented with:

Technical-only XGBoost
XGBoost + LLM features

The final project should use:

XGBoost + LLM

as the primary experimental model.

12. Prediction

There are two prediction concepts.

Historical test predictions

prediction.py

Used to evaluate the model against known historical test data.

Output:

Date
Close
Target_Return
Predicted_Return
Predicted_Next_Close
Signal

Future prediction

predict_future.py

Loads:

models/xgboost_llm.pkl

and uses the latest available market row.

It produces:

Current price
Predicted return
Predicted price
Signal

Signal thresholds currently follow:

prediction_return >  1%  → BUY
prediction_return < -1%  → SELL
otherwise                → HOLD

These thresholds should be treated as configurable research parameters, not investment recommendations.

13. Important Limitation for Future Dates

The model cannot directly predict August 12 or August 13, 2026 from historical data unless the required input features for those prediction dates are available.

For example, if the latest downloaded market data ends on:

2026-08-10

then:

2026-08-11
2026-08-12
2026-08-13

are future trading dates relative to the dataset.

The model needs current market/news inputs to generate a new prediction.

For each new trading day:

Latest market data
        +
Latest relevant news
        ↓
Technical features
        +
RAG + Ollama features
        ↓
XGBoost
        ↓
Next-day prediction

14. Recommended One-Command Pipeline

Create:

src/pipeline.py

The goal is to run the project in one command:

python src/pipeline.py

Recommended sequence:

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

15. Recommended Pipeline Design

pipeline.py should conceptually execute:

run(
    "data_download.py"
)

run(
    "data_preprocessing.py"
)

run(
    "feature_engineering.py"
)

run(
    "news_collection.py"
)

run(
    "news_preprocessing.py"
)

run(
    "build_faiss_index.py"
)

run(
    "generate_llm_features.py"
)

run(
    "merge_llm_features.py"
)

run(
    "llm_data_splitting.py"
)

run(
    "xgboost_training.py"
)

run(
    "prediction.py"
)

run(
    "predict_future.py"
)

Use subprocess.run(..., check=True) so that the pipeline stops immediately when a step fails.

16. Files That Can Be Removed or Deprecated

Do not delete files immediately. Move old/duplicate experiments into an archive/ directory first.

Recommended candidates:

split_llm_data.py

If:

llm_data_splitting.py

already performs the final LLM split correctly, keep only one splitting implementation.

Old experimental files such as:

generate_llm_features.py
llm_features.py
llm_sentiment.py

can be consolidated if they duplicate functionality.

Recommended final design:

generate_llm_features.py

should be the single entry point for LLM feature generation.

If both exist:

build_faiss_index.py
rag_index.py

use:

build_faiss_index.py

as the executable pipeline step and keep rag_index.py as the reusable RAG implementation module.

If both exist:

data_splitting.py
llm_data_splitting.py

keep both only if they have different responsibilities:

data_splitting.py
    → technical-only dataset

llm_data_splitting.py
    → final technical + LLM dataset

Otherwise consolidate them.

17. Do Not Delete These

Keep:

baseline.py
data_download.py
data_preprocessing.py
feature_engineering.py
news_collection.py
news_preprocessing.py
ollama_llm.py
rag_index.py
rag_retriever.py
build_faiss_index.py
generate_llm_features.py
merge_llm_features.py
llm_data_splitting.py
model_training.py
random_forest_training.py
xgboost_training.py
prediction.py
predict_future.py
pipeline.py

These represent the main project workflow and evaluation components.

18. Recommended Execution Order

First-time setup

python -m venv .venv

Activate:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Verify Ollama:

ollama list

Run the entire project

Once pipeline.py is implemented:

python src/pipeline.py

This should be the normal project execution method.

19. Manual Execution

If debugging individual stages:

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

20. RAG + Ollama Data Flow

The intended production/research flow is:

NVDA market data
        │
        ├──────────────► Technical features
        │
        │
Financial news
        │
        ▼
News preprocessing
        │
        ▼
FAISS vector index
        │
        ▼
Retrieve relevant news
        │
        ▼
Ollama LLM
        │
        ▼
Structured JSON
        │
        ├── LLM_Sentiment
        ├── LLM_Market_Impact
        ├── LLM_AI_Demand
        ├── LLM_Regulatory_Risk
        ├── LLM_Earnings_Outlook
        ├── LLM_Supply_Chain_Risk
        ├── LLM_Competition_Risk
        └── LLM_Confidence
        │
        ▼
Merge with technical features
        │
        ▼
XGBoost
        │
        ▼
Next-day return
        │
        ▼
Predicted price + signal

21. Data Leakage Prevention

This project must avoid future information entering the training features.

Rules:

Sort all data chronologically.

Create lagged features only from past data.

Create Target_Return using the next trading day's close.

Do not use Next_Close as an input feature.

Split chronologically.

News used for a prediction must have been published before the prediction timestamp.

Do not generate LLM features using future news.

Do not fit preprocessing/scalers on the complete dataset before splitting.

Keep the test set untouched until final evaluation.

22. Why the LLM Does Not Automatically Improve Accuracy

The LLM is useful for information that technical indicators cannot directly represent, such as:

NVIDIA earnings expectations

AI infrastructure demand

GPU demand

regulatory developments

supply-chain disruptions

competitive announcements

major customer announcements

sentiment around NVIDIA

However:

LLM ≠ guaranteed prediction accuracy

LLM features are noisy and may add no predictive value.

The correct experiment is:

Model A:
Technical features → XGBoost

vs.

Model B:
Technical features + LLM features → XGBoost

Compare both on the exact same unseen test period.

23. Recommended Evaluation Metrics

Do not rely only on price R².

Use:

Return MAE
Return RMSE
Return R²
Directional Accuracy
Precision
Recall
F1
Maximum Drawdown
Sharpe Ratio

For a trading-oriented project, directional accuracy and backtested risk-adjusted performance are especially important.

24. Current Project Status

Current successful pipeline components include:

✓ Historical NVIDIA data
✓ Technical feature engineering
✓ Chronological train/validation/test split
✓ Naive baseline
✓ Linear Regression
✓ Random Forest
✓ XGBoost
✓ Ollama integration
✓ LLM financial features
✓ LLM + XGBoost
✓ Future prediction script
✓ RAG architecture
✓ FAISS architecture

The main remaining engineering task is to make:

src/pipeline.py

the single orchestration entry point.

25. Final Recommended Workflow

                ┌─────────────────┐
                │ pipeline.py     │
                └────────┬────────┘
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
       ▼                 ▼                  ▼
 Market Data          News Data          Ollama
       │                 │                  │
       ▼                 ▼                  │
 Technical Features   RAG/FAISS ◄──────────┘
       │                 │
       └────────┬────────┘
                ▼
       Combined Dataset
                │
                ▼
       Train / Validation / Test
                │
                ▼
        XGBoost + LLM
                │
                ▼
          Evaluation
                │
                ▼
       Future Prediction
                │
                ▼
       Return / Price / Signal

26. Example Final Output

============================================================
NVIDIA FUTURE PREDICTION
============================================================

Latest market date : 2026-08-10
Current price      : $217.55

Predicted return   : 0.0770%
Predicted price    : $217.72

Signal             : HOLD

The prediction is a model output and should not be interpreted as a guaranteed future price.

27. Project Goal

The final objective is to build a reproducible hybrid financial prediction system:

Technical Analysis
        +
Financial News
        +
RAG
        +
Local Ollama LLM
        +
XGBoost
        ↓
Next-Day NVIDIA Return Prediction

The architecture is designed so that the LLM acts as an information-extraction and contextual-signal layer, while XGBoost remains the primary numerical prediction model.
