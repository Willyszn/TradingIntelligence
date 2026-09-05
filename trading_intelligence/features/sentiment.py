from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_sentiment(events: pd.DataFrame, bar_index: pd.DataFrame) -> pd.DataFrame:
    """Causally aggregate event features to bars; events after a bar cannot influence it."""
    required = {"timestamp", "asset", "sentiment"}
    if not required.issubset(events.columns):
        raise ValueError(f"Events need columns: {sorted(required)}")

    e = events.copy()
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
    e = e.dropna(subset=["sentiment"]).sort_values("timestamp")
    b = bar_index.copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)

    rows: list[dict] = []
    for symbol, bars in b.groupby("symbol", sort=False):
        ev = e[e["asset"] == symbol].copy()
        for _, bar in bars.sort_values("timestamp").iterrows():
            prior = ev[ev["timestamp"] <= bar["timestamp"]]
            if prior.empty:
                rows.append({"timestamp": bar["timestamp"], "symbol": symbol,
                             "sentiment_mean": np.nan, "sentiment_ewm": np.nan,
                             "sentiment_count": 0, "sentiment_shock": np.nan,
                             "event_surprise": np.nan, "event_novelty": np.nan,
                             "event_relevance": np.nan, "source_credibility": np.nan})
                continue
            recent = prior.tail(20)
            current = float(recent["sentiment"].iloc[-1])
            hist_mean = float(recent["sentiment"].mean())
            rows.append({
                "timestamp": bar["timestamp"],
                "symbol": symbol,
                "sentiment_mean": current,
                "sentiment_ewm": float(prior["sentiment"].ewm(span=min(10, len(prior))).mean().iloc[-1]),
                "sentiment_count": int(len(recent)),
                "sentiment_shock": current - hist_mean,
                "event_surprise": float(recent["surprise"].dropna().mean()) if "surprise" in recent else np.nan,
                "event_novelty": float(recent["novelty"].dropna().mean()) if "novelty" in recent else np.nan,
                "event_relevance": float(recent["relevance"].dropna().mean()) if "relevance" in recent else np.nan,
                "source_credibility": float(recent["credibility"].dropna().mean()) if "credibility" in recent else np.nan,
            })
    return pd.DataFrame(rows)


def add_sentiment_features(frame: pd.DataFrame, sentiment_frame: pd.DataFrame) -> pd.DataFrame:
    """Join already-computed causal sentiment features to market bars."""
    out = frame.copy()
    s = sentiment_frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
    return out.merge(s, on=["timestamp", "symbol"], how="left")
