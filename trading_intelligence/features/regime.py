from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    out = df.sort_values(["symbol", "timestamp"]).copy()
    g = out.groupby("symbol", group_keys=False)
    out["trend_strength"] = g["close"].transform(
        lambda s: s.pct_change(20) / s.pct_change().rolling(20, min_periods=20).std()
    )
    out["realized_vol"] = g["close"].transform(
        lambda s: np.log(s).diff().rolling(vol_window, min_periods=vol_window).std() * np.sqrt(vol_window)
    )
    out["vol_regime_z"] = g["realized_vol"].transform(
        lambda s: (s - s.rolling(60, min_periods=60).mean()) / s.rolling(60, min_periods=60).std()
    )
    out["regime_label"] = np.select(
        [out["trend_strength"] > 1.0, out["trend_strength"] < -1.0],
        ["bull_trend", "bear_trend"],
        default="range_or_unclear",
    )
    return out
