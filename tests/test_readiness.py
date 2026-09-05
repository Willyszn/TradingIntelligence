import pandas as pd
import pytest

from trading_intelligence.research.readiness import assess_market_dataset


def _frame(symbols=("AAA", "BBB"), rows=10):
    ts = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    out = []
    for symbol in symbols:
        for t in ts:
            out.append({"timestamp": t, "symbol": symbol, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
    return pd.DataFrame(out)


def test_readiness_warns_on_small_dataset():
    result = assess_market_dataset(_frame(), min_rows_per_symbol=20, min_symbols=3)
    assert result.status == "NOT_READY"
    assert result.symbols == 2
    assert result.min_rows_per_symbol == 10


def test_readiness_ready_when_thresholds_met():
    result = assess_market_dataset(_frame(symbols=tuple("ABCDEF"), rows=800), min_rows_per_symbol=750, min_symbols=5)
    assert result.status == "READY"
    assert result.min_rows_per_symbol == 800


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        assess_market_dataset(pd.DataFrame({"timestamp": [], "symbol": []}))
