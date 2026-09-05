import pandas as pd

from trading_intelligence.features.sentiment import aggregate_sentiment


def test_sentiment_is_causal():
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 11:00"], utc=True),
        "symbol": ["XYZ", "XYZ"],
    })
    events = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 10:30", "2025-01-01 12:00"], utc=True),
        "asset": ["XYZ", "XYZ"], "sentiment": [0.8, -0.9],
        "surprise": [0.2, -0.7], "novelty": [0.9, 0.9], "relevance": [1.0, 1.0], "credibility": [1.0, 1.0]
    })
    out = aggregate_sentiment(events, bars).sort_values("timestamp")
    assert pd.isna(out.iloc[0]["sentiment_mean"])
    assert abs(out.iloc[1]["sentiment_mean"] - 0.8) < 1e-9
