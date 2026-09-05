import pandas as pd
import pytest

from trading_intelligence.data.validation import validate_ohlcv


def test_validation_rejects_bad_ohlc():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01"], utc=True),
        "symbol": ["XYZ"], "open": [10], "high": [8], "low": [7], "close": [9]
    })
    with pytest.raises(ValueError):
        validate_ohlcv(df)
