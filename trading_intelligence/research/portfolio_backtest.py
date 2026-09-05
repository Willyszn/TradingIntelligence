from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    cost_bps_per_side: float = 5.0
    signal_threshold: float = 0.0
    holding_bars: int = 3
    max_positions: int | None = None
    allow_short: bool = True


def run_equal_weight_event_portfolio(
    frame: pd.DataFrame,
    prediction_column: str = "expected_return",
    config: PortfolioBacktestConfig | None = None,
) -> pd.DataFrame:
    """Simulate equal-weight entries at the next bar open and hold N bars.

    Each decision timestamp receives equal capital across the selected signals.
    The output is one portfolio return per decision timestamp, avoiding the
    misleading practice of treating every asset-trade row as an independent
    daily portfolio observation.
    """
    cfg = config or PortfolioBacktestConfig()
    required = {"symbol", "timestamp", prediction_column, "open", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Portfolio backtest missing columns: {sorted(missing)}")
    if cfg.holding_bars < 1:
        raise ValueError("holding_bars must be >= 1")

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)
    df["entry_timestamp"] = g["timestamp"].shift(-1)
    df["entry_price"] = g["open"].shift(-1)
    df["exit_timestamp"] = g["timestamp"].shift(-cfg.holding_bars)
    df["exit_price"] = g["close"].shift(-cfg.holding_bars)
    df["trade_return"] = df["exit_price"] / df["entry_price"] - 1.0

    pred = pd.to_numeric(df[prediction_column], errors="coerce")
    df["signal"] = np.where(pred > cfg.signal_threshold, 1.0, np.where(pred < -cfg.signal_threshold, -1.0, 0.0))
    if not cfg.allow_short:
        df["signal"] = np.clip(df["signal"], 0.0, 1.0)

    df = df.dropna(subset=["entry_timestamp", "entry_price", "exit_price", "trade_return"])
    rows: list[dict[str, object]] = []
    for ts, grp in df.groupby("timestamp", sort=True):
        active = grp[grp["signal"] != 0].copy()
        if active.empty:
            rows.append({
                "timestamp": ts,
                "portfolio_return": 0.0,
                "gross_return": 0.0,
                "cost": 0.0,
                "positions": 0,
                "long_positions": 0,
                "short_positions": 0,
            })
            continue
        active["abs_pred"] = active[prediction_column].astype(float).abs()
        if cfg.max_positions is not None and len(active) > cfg.max_positions:
            active = active.nlargest(cfg.max_positions, "abs_pred")
        # Equal capital per selected trade; signed by signal direction.
        gross = float((active["signal"] * active["trade_return"]).mean())
        cost = float(2.0 * cfg.cost_bps_per_side / 10_000.0)
        rows.append({
            "timestamp": ts,
            "portfolio_return": gross - cost,
            "gross_return": gross,
            "cost": cost,
            "positions": int(len(active)),
            "long_positions": int((active["signal"] > 0).sum()),
            "short_positions": int((active["signal"] < 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
