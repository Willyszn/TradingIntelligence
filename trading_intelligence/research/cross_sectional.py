from __future__ import annotations

import numpy as np
import pandas as pd


def rank_cross_section(frame: pd.DataFrame, score_column: str = "expected_return") -> pd.DataFrame:
    """Rank securities independently at each timestamp, avoiding time-series leakage."""
    required = {"timestamp", "symbol", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = frame.copy()
    out["cross_sectional_rank"] = out.groupby("timestamp")[score_column].rank(pct=True, method="average")
    out["cross_sectional_z"] = out.groupby("timestamp")[score_column].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan)
    )
    return out


def top_bottom_portfolio(frame: pd.DataFrame, score_column: str = "expected_return", quantile: float = 0.2) -> pd.DataFrame:
    """Create a simple market-neutral research portfolio from cross-sectional scores."""
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    ranked = rank_cross_section(frame, score_column=score_column)
    long = ranked[ranked["cross_sectional_rank"] >= 1 - quantile].copy()
    short = ranked[ranked["cross_sectional_rank"] <= quantile].copy()
    long["portfolio_weight"] = 1.0
    short["portfolio_weight"] = -1.0
    selected = pd.concat([long, short], ignore_index=True)
    counts = selected.groupby(["timestamp", "portfolio_weight"])["symbol"].transform("count")
    selected["portfolio_weight"] = selected["portfolio_weight"] / counts.replace(0, np.nan)
    return selected.sort_values(["timestamp", "portfolio_weight"], ascending=[True, False]).reset_index(drop=True)
