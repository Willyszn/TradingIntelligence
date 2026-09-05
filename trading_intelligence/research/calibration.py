from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: pd.Series | np.ndarray, p_positive: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_positive, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and p_positive must have the same shape")
    return float(np.mean((p - y) ** 2))


def probability_bins(p_positive: pd.Series, y_true: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Calibration table; probabilities are never interpreted as guaranteed outcomes."""
    x = pd.DataFrame({"p": p_positive.astype(float), "y": y_true.astype(float)}).dropna()
    if x.empty:
        return pd.DataFrame(columns=["bin", "count", "mean_pred", "actual_rate"])
    edges = np.linspace(0, 1, bins + 1)
    x["bin"] = pd.cut(x["p"], bins=edges, include_lowest=True, duplicates="drop")
    out = x.groupby("bin", observed=False).agg(count=("y", "size"), mean_pred=("p", "mean"), actual_rate=("y", "mean")).reset_index()
    return out
