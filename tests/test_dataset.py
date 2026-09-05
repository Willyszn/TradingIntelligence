import pandas as pd

from trading_intelligence.research.dataset import build_supervised_market_dataset, join_point_in_time_news


def sample_market():
    t = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": t,
        "open": range(10, 18),
        "high": [11,12,13,14,15,16,17,18],
        "low": [9,10,11,12,13,14,15,16],
        "close": range(10, 18),
        "volume": [100]*8,
        "symbol": ["AAPL"]*8,
    })


def test_forward_target_is_future_and_features_are_causal():
    df = build_supervised_market_dataset(sample_market(), horizon=3)
    assert df.loc[0, "forward_return"] == (13 / 11) - 1
    assert pd.isna(df.loc[7, "forward_return"])


def test_news_join_excludes_future_news():
    m = sample_market().iloc[[3]].copy()
    news = pd.DataFrame([
        {"timestamp": "2025-01-04T00:00:00Z", "asset": "AAPL", "sentiment": 1.0, "sentiment_change": .5, "surprise": .2},
        {"timestamp": "2025-01-05T00:00:00Z", "asset": "AAPL", "sentiment": -1.0, "sentiment_change": -1.0, "surprise": -1.0},
    ])
    out = join_point_in_time_news(m, news, lookback_hours=24)
    assert out.iloc[0]["news_count_24h"] == 1
    assert out.iloc[0]["sentiment_mean_24h"] == 1.0


def test_next_bar_entry_target_uses_next_open_and_horizon_close():
    df = build_supervised_market_dataset(sample_market(), horizon=3)
    # Signal on day 0 -> enter day 1 open (11) -> exit day 3 close (13).
    assert df.loc[0, "entry_price"] == 11
    assert df.loc[0, "forward_return"] == (13 / 11) - 1
    assert df.loc[0, "entry_timestamp"] == df.loc[1, "timestamp"]
    assert df.loc[0, "target_timestamp"] == df.loc[3, "timestamp"]
