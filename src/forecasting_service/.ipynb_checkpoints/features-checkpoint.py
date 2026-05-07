from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays
except Exception:  # pragma: no cover - optional dependency at import time
    holidays = None


LAG_PERIODS = (1, 7, 30)
ROLLING_WINDOWS = (4, 8)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["year"] = out["date"].dt.year

    if holidays is not None:
        us_holidays = holidays.US(years=range(out["year"].min(), out["year"].max() + 2))
        out["holiday_flag"] = out["date"].dt.date.isin(us_holidays).astype(int)
    else:
        out["holiday_flag"] = 0
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["state", "date"]).copy()
    grouped = out.groupby("state")["sales"]
    for lag in LAG_PERIODS:
        out[f"lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = grouped.shift(1)
        out[f"rolling_mean_{window}"] = shifted.groupby(out["state"]).rolling(window).mean().reset_index(level=0, drop=True)
        out[f"rolling_std_{window}"] = shifted.groupby(out["state"]).rolling(window).std().reset_index(level=0, drop=True)
    return out


def make_supervised_features(df: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    out = add_calendar_features(add_lag_features(df))
    out = out.replace([np.inf, -np.inf], np.nan)
    if drop_missing:
        out = out.dropna().reset_index(drop=True)
    return out


def feature_columns() -> list[str]:
    columns = [
        "day_of_week",
        "month",
        "quarter",
        "week_of_year",
        "year",
        "holiday_flag",
    ]
    columns.extend([f"lag_{lag}" for lag in LAG_PERIODS])
    for window in ROLLING_WINDOWS:
        columns.extend([f"rolling_mean_{window}", f"rolling_std_{window}"])
    return columns


def next_week_rows(history: pd.DataFrame, horizon: int, freq: str = "W-SUN") -> pd.DataFrame:
    state = history["state"].iloc[0]
    last_date = history["date"].max()
    future_dates = pd.date_range(last_date + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
    return pd.DataFrame({"state": state, "date": future_dates, "sales": np.nan})
