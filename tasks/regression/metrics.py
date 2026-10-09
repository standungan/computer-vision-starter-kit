"""Regression metrics, all in the target's original units.

    MAE   mean absolute error: "on average we are off by this much"
    RMSE  root mean squared error: like MAE but punishes big misses harder
    R2    1 = perfect, 0 = no better than always predicting the mean, <0 = worse than that
"""
from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0}
