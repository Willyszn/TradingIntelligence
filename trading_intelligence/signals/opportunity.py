from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_score(value: float, low: float, high: float) -> float:
    if not np.isfinite(value) or high <= low:
        return 50.0
    return float(np.clip((value - low) / (high - low) * 100.0, 0.0, 100.0))


def directional_score(expected_return: float, volatility: float | None = None) -> float:
    if not np.isfinite(expected_return):
        return 50.0
    if volatility is None or not np.isfinite(volatility) or volatility <= 0:
        return float(np.clip(50 + expected_return * 1000, 0, 100))
    z = expected_return / volatility
    return float(50 + 50 * np.tanh(z))


def combine_scores(quant_score: float, sentiment_score: float | None = None,
                   event_score: float | None = None, regime_score: float | None = None) -> float:
    values = [quant_score]
    values.extend(v for v in (sentiment_score, event_score, regime_score) if v is not None and np.isfinite(v))
    return float(np.mean(values))


def rank_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert model outputs into an interpretable cross-sectional opportunity ranking."""
    out = frame.copy()
    if "expected_return" not in out.columns:
        raise ValueError("expected_return is required")
    out["direction"] = np.where(out["expected_return"] >= 0, "LONG", "SHORT")
    if "volatility_20" in out.columns:
        out["risk_adjusted_alpha"] = out["expected_return"] / out["volatility_20"].replace(0, np.nan)
    else:
        out["risk_adjusted_alpha"] = out["expected_return"]
    q = directional_score(out["expected_return"].median(), out.get("volatility_20", pd.Series([np.nan])).median())
    out["quant_score"] = np.clip(50 + (out["expected_return"] / (out["expected_return"].abs().quantile(0.8) + 1e-9)) * 50, 0, 100)
    if "sentiment_mean" in out.columns:
        out["sentiment_score"] = np.clip(50 + 50 * out["sentiment_mean"].fillna(0), 0, 100)
    if "event_surprise" in out.columns:
        out["event_score"] = np.clip(50 + 50 * out["event_surprise"].fillna(0), 0, 100)
    if "trend_strength" in out.columns:
        out["regime_score"] = np.clip(50 + 25 * np.tanh(out["trend_strength"].fillna(0)), 0, 100)
    out["opportunity_score"] = out.apply(
        lambda r: combine_scores(r["quant_score"], r.get("sentiment_score"), r.get("event_score"), r.get("regime_score")),
        axis=1,
    )
    out["model_context_score"] = q
    return out.sort_values(["opportunity_score", "risk_adjusted_alpha"], ascending=False)
