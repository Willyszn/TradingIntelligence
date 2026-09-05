import pandas as pd

from trading_intelligence.signals.opportunity import rank_opportunities


def test_rank_opportunities():
    frame = pd.DataFrame({
        "symbol": ["A", "B"],
        "expected_return": [0.02, -0.01],
        "volatility_20": [0.01, 0.02],
        "sentiment_mean": [0.8, -0.5],
        "event_surprise": [0.4, -0.2],
        "trend_strength": [1.2, -0.5],
    })
    out = rank_opportunities(frame)
    assert out.iloc[0]["symbol"] == "A"
    assert set(out["direction"]) == {"LONG", "SHORT"}
