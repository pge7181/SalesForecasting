from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from forecasting_service.features import feature_columns, make_supervised_features, next_week_rows


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> "Forecaster":
        ...

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        ...


@dataclass
class ForecastResult:
    model_name: str
    forecast: pd.DataFrame
    smape: float
    rmse: float
    mae: float


class SarimaForecaster:
    name = "SARIMA"

    def __init__(self, seasonal_periods: int = 52):
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, train: pd.DataFrame) -> "SarimaForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = train.sort_values("date").set_index("date")["sales"].asfreq("W-SUN")
        seasonal_order = (1, 1, 1, self.seasonal_periods) if len(series) >= self.seasonal_periods * 2 else (0, 0, 0, 0)
        self.model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        future = next_week_rows(history, horizon)
        forecast = self.model.forecast(steps=horizon)
        future["prediction"] = np.maximum(np.asarray(forecast), 0)
        return future[["date", "prediction"]]


class ProphetForecaster:
    name = "Prophet"

    def __init__(self):
        self.model = None

    def fit(self, train: pd.DataFrame) -> "ProphetForecaster":
        from prophet import Prophet

        prophet_df = train.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        self.model.fit(prophet_df)
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        future_dates = next_week_rows(history, horizon).rename(columns={"date": "ds"})
        forecast = self.model.predict(future_dates[["ds"]])
        out = pd.DataFrame({"date": future_dates["ds"], "prediction": np.maximum(forecast["yhat"].to_numpy(), 0)})
        return out


class XGBoostForecaster:
    name = "XGBoost"

    def __init__(self):
        self.model = None

    def fit(self, train: pd.DataFrame) -> "XGBoostForecaster":
        from xgboost import XGBRegressor

        supervised = make_supervised_features(train)
        self.model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self.model.fit(supervised[feature_columns()], supervised["sales"])
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        working = history.sort_values("date").copy()
        predictions = []
        for _ in range(horizon):
            candidate = pd.concat([working, next_week_rows(working, 1)], ignore_index=True)
            supervised = make_supervised_features(candidate, drop_missing=False).tail(1)
            pred = float(self.model.predict(supervised[feature_columns()])[0])
            pred = max(pred, 0.0)
            forecast_date = supervised["date"].iloc[0]
            predictions.append({"date": forecast_date, "prediction": pred})
            working = pd.concat(
                [working, pd.DataFrame({"state": [working["state"].iloc[0]], "date": [forecast_date], "sales": [pred]})],
                ignore_index=True,
            )
        return pd.DataFrame(predictions)


class LSTMForecaster:
    name = "LSTM"

    def __init__(self, lookback: int = 30, epochs: int = 25):
        self.lookback = lookback
        self.epochs = epochs
        self.model = None
        self.mean = 0.0
        self.std = 1.0

    def fit(self, train: pd.DataFrame) -> "LSTMForecaster":
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        values = train.sort_values("date")["sales"].astype(float).to_numpy()
        self.mean = float(values.mean())
        self.std = float(values.std() or 1.0)
        scaled = (values - self.mean) / self.std
        x, y = self._make_sequences(scaled)

        model = Sequential(
            [
                LSTM(48, input_shape=(self.lookback, 1)),
                Dropout(0.15),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(x, y, epochs=self.epochs, batch_size=16, verbose=0)
        self.model = model
        return self

    def _make_sequences(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for i in range(self.lookback, len(values)):
            x.append(values[i - self.lookback : i])
            y.append(values[i])
        return np.asarray(x).reshape(-1, self.lookback, 1), np.asarray(y)

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        working = history.sort_values("date").copy()
        values = working["sales"].astype(float).to_list()
        rows = []
        for forecast_date in next_week_rows(working, horizon)["date"]:
            window = np.asarray(values[-self.lookback :], dtype=float)
            scaled = ((window - self.mean) / self.std).reshape(1, self.lookback, 1)
            pred_scaled = float(self.model.predict(scaled, verbose=0)[0][0])
            pred = max(pred_scaled * self.std + self.mean, 0.0)
            values.append(pred)
            rows.append({"date": forecast_date, "prediction": pred})
        return pd.DataFrame(rows)


def available_models(skip_lstm: bool = False) -> list[Forecaster]:
    models: list[Forecaster] = [
        SarimaForecaster(),
        ProphetForecaster(),
        XGBoostForecaster(),
    ]
    if not skip_lstm:
        models.append(LSTMForecaster())
    return models
