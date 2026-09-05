from __future__ import annotations

import numpy as np
import pandas as pd

def naive_zero_predictions(index: pd.Index) -> np.ndarray:
    return np.zeros(len(index), dtype=float)

def sign_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    yt = np.sign(y_true.to_numpy())
    yp = np.sign(np.asarray(y_pred))
    return float(np.mean(yt == yp)) if len(yt) else float("nan")

def bucket_summary(y_true: pd.Series, y_pred: np.ndarray, q: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y_true.to_numpy(), "pred": np.asarray(y_pred)})
    if frame.empty:
        return frame
    try:
        frame["bucket"] = pd.qcut(frame["pred"], q=q, labels=False, duplicates="drop")
    except ValueError:
        frame["bucket"] = 0
    return frame.groupby("bucket", observed=True)["y"].agg(["count", "mean", "median"]).reset_index()
