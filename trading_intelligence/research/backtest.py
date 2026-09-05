from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps_per_side: float = 5.0
    signal_threshold: float = 0.0
    holding_bars: int = 3
    allow_short: bool = True
    entry_timing: str = "next_open"


def run_signal_backtest(frame: pd.DataFrame, prediction_column: str = "expected_return", config: BacktestConfig | None = None) -> pd.DataFrame:
    """Backtest signal-at-close decisions with next-bar-open entry.

    A horizon of N means entry at t+1 open and exit at the close of t+N.
    """
    cfg = config or BacktestConfig()
    if cfg.entry_timing != "next_open":
        raise ValueError("Only next_open entry timing is supported")
    if cfg.holding_bars < 1:
        raise ValueError("holding_bars must be >= 1")
    required = {"symbol", "timestamp", prediction_column, "open", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Backtest frame missing columns: {sorted(missing)}")

    out = frame.sort_values(["symbol", "timestamp"]).copy()
    g = out.groupby("symbol", group_keys=False)
    out["entry_timestamp"] = g["timestamp"].shift(-1)
    out["entry_price"] = g["open"].shift(-1)
    out["exit_timestamp"] = g["timestamp"].shift(-cfg.holding_bars)
    out["exit_price"] = g["close"].shift(-cfg.holding_bars)
    out["future_return"] = out["exit_price"] / out["entry_price"] - 1.0

    pred = out[prediction_column].astype(float)
    signal = np.where(pred > cfg.signal_threshold, 1.0, np.where(pred < -cfg.signal_threshold, -1.0, 0.0))
    if not cfg.allow_short:
        signal = np.clip(signal, 0, 1)
    out["signal"] = signal
    out["gross_return"] = out["signal"] * out["future_return"]
    traded = out["signal"] != 0
    out["cost"] = np.where(traded, 2.0 * cfg.cost_bps_per_side / 10_000.0, 0.0)
    out["net_return"] = out["gross_return"] - out["cost"]
    return out.dropna(subset=["entry_price", "exit_price", "future_return"]).reset_index(drop=True)
