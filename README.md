# End-to-End State Sales Forecasting Service

This project trains a production-style time series forecasting system for weekly state sales and exposes the next 8 weeks of predictions through a REST API.

## What It Does

- Loads and cleans the sales dataset from `data/sales_history.csv`
- Parses dates as `day/month/year`, which makes the data a clean weekly series
- Fills missing weekly dates per state and imputes missing sales values
- Builds lag, rolling, calendar, and holiday features
- Trains and compares four model families:
  - SARIMA
  - Prophet
  - XGBoost with lag features
  - LSTM
- Uses a time-based validation split, so future data never leaks into training
- Selects the best model per state using validation SMAPE
- Saves forecasts and metrics to `artifacts/model_registry.joblib`
- Serves predictions through FastAPI

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Train

```bash
python -m forecasting_service.training --data "/Users/pge7181/Desktop/New project/data/sales_history.csv" --horizon 8
```

Useful faster smoke test:

```bash
python -m forecasting_service.training --data "/Users/pge7181/Desktop/New project/data/sales_history.csv" --horizon 8
```

## Run API

```bash
python -m uvicorn forecasting_service.api:app --reload
```

Then open:

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/states`
- `GET http://127.0.0.1:8000/forecast/California?horizon=8`
- `GET http://127.0.0.1:8000/models`

## High-Level Explanation

The dataset has one weekly sales value per state. The pipeline first cleans numeric sales, parses dates correctly, aggregates duplicate rows if they ever appear, and creates a complete weekly calendar for each state. If any week is missing, it is inserted and sales are imputed from nearby values.

The validation split is time-based: the last 8 weeks are held out and models train only on earlier weeks. This mirrors the real forecasting problem, because a model in production never gets to see future weeks while learning.

Feature engineering gives machine learning models historical context. Lag features such as `lag_1`, `lag_7`, and `lag_30` represent prior sales from 1, 7, and 30 weekly periods ago. Rolling mean and rolling standard deviation summarize recent momentum and volatility. Calendar fields and holiday flags help the models learn seasonal behavior.

SARIMA and Prophet model trend and seasonality directly. XGBoost learns from engineered lag and calendar features. LSTM learns patterns from ordered sequences of previous sales values. The trainer evaluates every model on the same validation window, ranks them by SMAPE, and stores the best forecast for each state.

The API is intentionally separated from training. Training is a batch job that creates an artifact. The FastAPI app loads that artifact at startup and serves forecasts quickly, just like a real backend service.
