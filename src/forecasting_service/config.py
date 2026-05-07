from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_path: Path = PROJECT_ROOT / "data" / "sales_history.csv"
    artifact_path: Path = PROJECT_ROOT / "artifacts" / "model_registry.joblib"
    forecast_horizon: int = 8
    validation_weeks: int = 8
    weekly_frequency: str = "W-SUN"
    seasonal_periods: int = 52


settings = Settings()
