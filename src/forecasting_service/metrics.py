from __future__ import annotations

import numpy as np


def smape(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual) + np.abs(pred)
    ratio = np.where(denominator == 0, 0.0, 2.0 * np.abs(pred - actual) / denominator)
    return float(np.mean(ratio) * 100)


def rmse(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mae(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(actual - pred)))
