from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from forecasting_service.config import settings
from forecasting_service.data import complete_weekly_series, load_sales_data, train_validation_split
from forecasting_service.metrics import mae, rmse, smape
from forecasting_service.models import ForecastResult, available_models


def evaluate_model(model, train: pd.DataFrame, validation: pd.DataFrame, horizon: int) -> ForecastResult:
    fitted = model.fit(train)
    forecast = fitted.predict(train, horizon)
    merged = validation[["date", "sales"]].merge(forecast, on="date", how="left")
    if merged["prediction"].isna().any():
        raise ValueError(f"{model.name} produced missing validation predictions.")
    return ForecastResult(
        model_name=model.name,
        forecast=forecast,
        smape=smape(merged["sales"], merged["prediction"]),
        rmse=rmse(merged["sales"], merged["prediction"]),
        mae=mae(merged["sales"], merged["prediction"]),
    )


def train_for_state(state_df: pd.DataFrame, horizon: int, skip_lstm: bool) -> dict:
    train, validation = train_validation_split(state_df, horizon)
    results = []
    errors = {}
    for model in available_models(skip_lstm=skip_lstm):
        try:
            results.append(evaluate_model(model, train, validation, horizon))
        except Exception as exc:
            errors[model.name] = str(exc)

    if not results:
        raise RuntimeError(f"Every model failed for {state_df['state'].iloc[0]}: {errors}")

    best = sorted(results, key=lambda item: item.smape)[0]
    final_model = next(model for model in available_models(skip_lstm=skip_lstm) if model.name == best.model_name)
    final_model.fit(state_df)
    future_forecast = final_model.predict(state_df, horizon)
    metrics = {
        item.model_name: {
            "smape": item.smape,
            "rmse": item.rmse,
            "mae": item.mae,
        }
        for item in results
    }
    return {
        "state": state_df["state"].iloc[0],
        "best_model": best.model_name,
        "metrics": metrics,
        "errors": errors,
        "forecast": [
            {"date": row.date.date().isoformat(), "prediction": float(row.prediction)}
            for row in future_forecast.itertuples(index=False)
        ],
    }


def train_all(
    data_path: Path,
    artifact_path: Path,
    horizon: int,
    selected_states: list[str] | None = None,
    skip_lstm: bool = False,
) -> dict:
    raw = load_sales_data(data_path)
    weekly = complete_weekly_series(raw)
    if selected_states:
        weekly = weekly[weekly["state"].isin(selected_states)]
    if weekly.empty:
        raise ValueError("No matching states found.")

    registry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "horizon": horizon,
        "source_data": str(data_path),
        "states": {},
    }
    for state, state_df in weekly.groupby("state", sort=True):
        print(f"Training {state}...")
        registry["states"][state] = train_for_state(state_df, horizon=horizon, skip_lstm=skip_lstm)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(registry, artifact_path)
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train state-level sales forecasting models.")
    parser.add_argument("--data", type=Path, default=settings.data_path)
    parser.add_argument("--artifact", type=Path, default=settings.artifact_path)
    parser.add_argument("--horizon", type=int, default=settings.forecast_horizon)
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM for faster local smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = train_all(args.data, args.artifact, args.horizon, args.states, args.skip_lstm)
    summary = {
        state: payload["best_model"]
        for state, payload in registry["states"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved artifact to {args.artifact}")


if __name__ == "__main__":
    main()
