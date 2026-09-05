from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from trading_intelligence.models.baseline import QuantBaselineModel


@dataclass(frozen=True)
class MultiMarketResult:
    market: str
    rows: int
    mean_return: float
    sharpe: float
    win_rate: float
    directional_accuracy: float


def fit_predict_oos(frame: pd.DataFrame, feature_columns: list[str], target: str = "target_return", train_fraction: float = 0.7) -> pd.DataFrame:
    """Chronological out-of-sample predictions, fitted separately per symbol."""
    required = {"symbol", "timestamp", target, *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = []
    for symbol, g in frame.sort_values("timestamp").groupby("symbol", sort=False):
        g = g.sort_values("timestamp").dropna(subset=feature_columns + [target]).reset_index(drop=True)
        if len(g) < 80:
            continue
        cut = int(len(g) * train_fraction)
        if cut < 50 or cut >= len(g):
            continue
        train, test = g.iloc[:cut], g.iloc[cut:].copy()
        model = QuantBaselineModel(random_state=42)
        model.fit(train, feature_columns, target)
        test["expected_return"] = model.predict(test)
        test["signal"] = np.sign(test["expected_return"])
        out.append(test)
    if not out:
        raise ValueError("No symbol has enough usable observations for out-of-sample prediction")
    return pd.concat(out, ignore_index=True)


def summarize_market(oos: pd.DataFrame) -> list[MultiMarketResult]:
    rows: list[MultiMarketResult] = []
    for symbol, g in oos.groupby("symbol"):
        actual = g["target_return"].astype(float)
        strat = g["signal"] * actual
        rows.append(MultiMarketResult(
            market=str(symbol),
            rows=len(g),
            mean_return=float(strat.mean()),
            sharpe=float((strat.mean()/strat.std(ddof=1))*np.sqrt(252)) if strat.std(ddof=1) else float("nan"),
            win_rate=float((strat > 0).mean()),
            directional_accuracy=float((np.sign(g["expected_return"]) == np.sign(actual)).mean()),
        ))
    return rows
