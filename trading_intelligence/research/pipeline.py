from __future__ import annotations

import pandas as pd

from trading_intelligence.features.quant import add_forward_target, add_quant_features
from trading_intelligence.features.regime import add_regime_features
from trading_intelligence.research.experiments import ExperimentResult, run_experiment


def prepare_research_frame(bars: pd.DataFrame, sentiment: pd.DataFrame | None = None,
                           horizon_bars: int = 3) -> pd.DataFrame:
    out = add_quant_features(bars)
    out = add_regime_features(out)
    out = add_forward_target(out, horizon_bars=horizon_bars)
    if sentiment is not None:
        from trading_intelligence.features.sentiment import add_sentiment_features
        out = add_sentiment_features(out, sentiment)
    return out


def candidate_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    quant = [c for c in frame.columns if c.startswith(("return_", "log_return_", "momentum_", "volatility_", "distance_sma_", "zscore_", "atr_", "volume_change_", "relative_volume_"))]
    regime = [c for c in ("trend_strength", "realized_vol", "vol_regime_z") if c in frame.columns]
    sentiment = [c for c in ("sentiment_mean", "sentiment_ewm", "sentiment_count", "sentiment_shock", "event_surprise", "event_novelty", "event_relevance", "source_credibility") if c in frame.columns]
    return {"quant_only": quant, "regime_aware": quant + regime, "sentiment_only": sentiment, "combined": quant + regime + sentiment}


def run_research_ladder(frame: pd.DataFrame, cost_bps_per_side: float = 5.0, horizon_bars: int = 3) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for name, cols in candidate_feature_sets(frame).items():
        if not cols:
            continue
        results.append(run_experiment(frame, name, cols, cost_bps_per_side=cost_bps_per_side, holding_bars=horizon_bars))
    return results
