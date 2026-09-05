from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(group: pd.DataFrame, window: int) -> pd.Series:
    prev_close = group["close"].shift(1)
    tr = pd.concat(
        [group["high"] - group["low"], (group["high"] - prev_close).abs(), (group["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def add_quant_features(df: pd.DataFrame, windows: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """Add causal price/volume features. All features use current/past information only."""
    out = df.sort_values(["symbol", "timestamp"]).copy()
    group = out.groupby("symbol", group_keys=False)
    out["return_1"] = group["close"].pct_change()
    out["log_return_1"] = group["close"].transform(lambda s: np.log(s).diff())

    for w in windows:
        out[f"momentum_{w}"] = group["close"].pct_change(w)
        out[f"volatility_{w}"] = group["log_return_1"].transform(
            lambda s, w=w: s.rolling(w, min_periods=w).std()
        )
        out[f"distance_sma_{w}"] = group["close"].transform(
            lambda s, w=w: s / s.rolling(w, min_periods=w).mean() - 1.0
        )
        out[f"zscore_{w}"] = group["close"].transform(
            lambda s, w=w: (s - s.rolling(w, min_periods=w).mean())
            / s.rolling(w, min_periods=w).std()
        )
        prev_close = group["close"].shift(1)
        true_range = pd.concat(
            [out["high"] - out["low"], (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        out[f"atr_{w}"] = true_range.groupby(out["symbol"]).transform(
            lambda s, w=w: s.rolling(w, min_periods=w).mean()
        )

    if "volume" in out.columns:
        out["volume_change_1"] = group["volume"].pct_change()
        out["relative_volume_20"] = group["volume"].transform(
            lambda s: s / s.rolling(20, min_periods=20).mean()
        )

    return out


def add_forward_target(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Create the canonical next-bar-open forward-return label.

    Signal is generated at the close of bar t. Entry is the open of t+1.
    For horizon N, exit is the close of t+N. The target therefore matches the
    next-bar execution convention used by the backtester.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    out = df.sort_values(["symbol", "timestamp"]).copy()
    g = out.groupby("symbol", group_keys=False)
    out["entry_timestamp"] = g["timestamp"].shift(-1)
    out["entry_price"] = g["open"].shift(-1)
    out["target_timestamp"] = g["timestamp"].shift(-horizon_bars)
    exit_close = g["close"].shift(-horizon_bars)
    out["target_return"] = exit_close / out["entry_price"] - 1.0
    # Backward-compatible alias for existing research utilities/tests.
    out["forward_return"] = out["target_return"]
    out["forward_positive"] = (out["target_return"] > 0).astype("Int64")
    return out
