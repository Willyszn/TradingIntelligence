import pandas as pd

from trading_intelligence.research.dataset import build_supervised_market_dataset


def test_crypto_dataset_builds_with_next_open_target():
    dates = pd.date_range("2021-01-01", periods=80, freq="D", tz="UTC")
    rows = []
    for j, symbol in enumerate(["BTCUSDT", "ETHUSDT"]):
        for i, ts in enumerate(dates):
            base = 100 + j * 20 + i * 0.5
            rows.append({"timestamp": ts, "open": base, "high": base+1, "low": base-1, "close": base+0.5, "volume": 1000+i, "symbol": symbol})
    df = pd.DataFrame(rows)
    out = build_supervised_market_dataset(df, horizon=3)
    valid = out.dropna(subset=["entry_price", "target_return"])
    first = valid.iloc[0]
    expected = valid.iloc[0]["entry_price"]
    assert first["entry_timestamp"] > first["timestamp"]
    assert expected > 0
