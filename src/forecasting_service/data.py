from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"State", "Date", "Total"}


def _clean_sales(value) -> float:
    if pd.isna(value):
        return np.nan
    return float(str(value).replace(",", "").strip())


def _parse_dayfirst_date(value) -> pd.Timestamp:
    text = str(value).strip()
    separator = "/" if "/" in text else "-"
    try:
        day, month, year = [int(part) for part in text.split(separator)]
        return pd.Timestamp(year=year, month=month, day=day)
    except Exception as exc:
        raise ValueError(f"Could not parse date value: {value}") from exc


def load_sales_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned["date"] = cleaned["Date"].map(_parse_dayfirst_date)
    cleaned["sales"] = cleaned["Total"].map(_clean_sales)
    cleaned["state"] = cleaned["State"].astype(str).str.strip()

    cleaned = cleaned.dropna(subset=["state", "sales"])
    cleaned = (
        cleaned.groupby(["state", "date"], as_index=False)["sales"]
        .sum()
        .sort_values(["state", "date"])
        .reset_index(drop=True)
    )
    return cleaned


def complete_weekly_series(df: pd.DataFrame, freq: str = "W-SUN") -> pd.DataFrame:
    completed = []
    for state, state_df in df.groupby("state", sort=True):
        state_df = state_df.sort_values("date").set_index("date")
        full_index = pd.date_range(state_df.index.min(), state_df.index.max(), freq=freq)
        reindexed = state_df.reindex(full_index)
        reindexed["state"] = state
        reindexed["sales"] = (
            reindexed["sales"]
            .astype(float)
            .interpolate(method="time")
            .ffill()
            .bfill()
        )
        reindexed = reindexed.rename_axis("date").reset_index()
        completed.append(reindexed[["state", "date", "sales"]])
    return pd.concat(completed, ignore_index=True).sort_values(["state", "date"])


def train_validation_split(
    state_df: pd.DataFrame,
    validation_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = state_df.sort_values("date").reset_index(drop=True)
    if len(ordered) <= validation_weeks + 30:
        raise ValueError("Not enough history for lag features and validation split.")
    return ordered.iloc[:-validation_weeks].copy(), ordered.iloc[-validation_weeks:].copy()
