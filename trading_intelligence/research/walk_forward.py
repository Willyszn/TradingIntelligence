from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from trading_intelligence.models.baseline import QuantBaselineModel
from trading_intelligence.research.metrics import summarize_strategy_returns


@dataclass
class FoldResult:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    rows: int
    mae: float
    r2: float
    directional_accuracy: float
    mean_strategy_return: float
    net_mean_strategy_return: float


def walk_forward_regression(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_size: int,
    test_size: int,
    step: int,
    random_state: int = 42,
    cost_bps_per_side: float = 0.0,
) -> list[FoldResult]:
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    usable = frame.dropna(subset=feature_columns + ["target_return"]).reset_index(drop=True)
    results: list[FoldResult] = []
    if train_size < 1 or test_size < 1 or step < 1:
        raise ValueError("train_size, test_size and step must be >= 1")

    start = 0
    while start + train_size + test_size <= len(usable):
        train = usable.iloc[start : start + train_size].copy()
        test = usable.iloc[start + train_size : start + train_size + test_size].copy()
        # Purge training rows whose forward-label window overlaps the test period.
        test_start = pd.Timestamp(test["timestamp"].min())
        if "target_timestamp" in train.columns:
            train = train[pd.to_datetime(train["target_timestamp"], utc=True) < test_start].copy()
        if len(train) < 50:
            start += step
            continue
        model = QuantBaselineModel(random_state=random_state)
        model.fit(train, feature_columns)
        pred = model.predict(test)
        actual = test["target_return"].to_numpy()
        direction = np.sign(pred)
        gross = direction * actual
        turnover = np.ones_like(gross)
        net = gross - turnover * (cost_bps_per_side * 2 / 10_000)
        results.append(FoldResult(
            train_end=pd.Timestamp(train["timestamp"].iloc[-1]),
            test_start=pd.Timestamp(test["timestamp"].iloc[0]),
            test_end=pd.Timestamp(test["timestamp"].iloc[-1]),
            rows=len(test),
            mae=float(mean_absolute_error(actual, pred)),
            r2=float(r2_score(actual, pred)) if np.var(actual) > 0 else float("nan"),
            directional_accuracy=float(np.mean(direction == np.sign(actual))),
            mean_strategy_return=float(np.mean(gross)),
            net_mean_strategy_return=float(np.mean(net)),
        ))
        start += step
    return results
