import numpy as np
import pandas as pd

from trading_intelligence.features.quant import add_forward_target
from trading_intelligence.research.dataset import build_supervised_market_dataset


def _bars():
    t = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": t,
        "symbol": ["A"] * len(t),
        "open": [10, 11, 12, 13, 14, 15],
        "high": [10, 11, 12, 13, 14, 15],
        "low": [10, 11, 12, 13, 14, 15],
        "close": [10, 11, 12, 13, 14, 15],
        "volume": [100] * len(t),
    })


def test_feature_target_matches_next_bar_open_convention():
    out = add_forward_target(_bars(), horizon_bars=3)
    first = out.iloc[0]
    assert first["entry_price"] == 11
    assert first["entry_timestamp"] == pd.Timestamp("2025-01-02", tz="UTC")
    assert first["target_timestamp"] == pd.Timestamp("2025-01-04", tz="UTC")
    assert np.isclose(first["target_return"], 13 / 11 - 1)


def test_assembly_target_matches_feature_target():
    out = build_supervised_market_dataset(_bars(), horizon=3)
    assert np.isclose(out.iloc[0]["target_return"], 13 / 11 - 1)
