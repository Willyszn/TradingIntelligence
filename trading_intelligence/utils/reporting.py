from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from trading_intelligence.research.walk_forward import FoldResult


def folds_to_frame(results: list[FoldResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def summarize_folds(results: list[FoldResult]) -> dict[str, float]:
    if not results:
        return {}
    frame = folds_to_frame(results)
    return {
        "folds": float(len(frame)),
        "mean_mae": float(frame["mae"].mean()),
        "mean_r2": float(frame["r2"].mean()),
        "mean_directional_accuracy": float(frame["directional_accuracy"].mean()),
        "mean_strategy_return": float(frame["mean_strategy_return"].mean()),
    }
