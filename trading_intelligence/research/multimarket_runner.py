from __future__ import annotations

from dataclasses import dataclass, asdict
import pandas as pd

from trading_intelligence.models.baseline import QuantBaselineModel
from trading_intelligence.research.backtest import BacktestConfig, run_signal_backtest
from trading_intelligence.research.metrics import summarize_strategy_returns
from trading_intelligence.research.splits import chronological_purged_split

@dataclass(frozen=True)
class MultiMarketResult:
    symbols: int
    rows: int
    mean_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    directional_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)

def pooled_time_split(frame: pd.DataFrame, train_end_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    return chronological_purged_split(frame, train_fraction=train_end_fraction)

def evaluate_pooled(frame: pd.DataFrame, feature_columns: list[str], cost_bps_per_side: float = 5.0, seed: int = 42) -> MultiMarketResult:
    required = {"timestamp", "symbol", "close", "target_return"} | set(feature_columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    usable = frame.dropna(subset=feature_columns + ["target_return"]).copy()
    train, test = pooled_time_split(usable)
    if len(train) < 50 or len(test) < 20:
        raise ValueError("Insufficient train/test observations after time split")
    model = QuantBaselineModel(random_state=seed)
    model.fit(train, feature_columns)
    test["expected_return"] = model.predict(test)
    bt = run_signal_backtest(test, config=BacktestConfig(cost_bps_per_side=cost_bps_per_side))
    summary = summarize_strategy_returns(bt["net_return"])
    direction = (test["expected_return"] > 0) == (test["target_return"] > 0)
    return MultiMarketResult(
        symbols=int(test["symbol"].nunique()), rows=len(bt),
        mean_return=summary.get("mean_return", float("nan")),
        sharpe=summary.get("sharpe", float("nan")),
        max_drawdown=summary.get("max_drawdown", float("nan")),
        win_rate=summary.get("win_rate", float("nan")),
        directional_accuracy=float(direction.mean()),
    )
