import pandas as pd
import numpy as np

from trading_intelligence.features.quant import add_forward_target, add_quant_features
from trading_intelligence.features.regime import add_regime_features


def _bars():
    ts = pd.date_range("2025-01-01", periods=80, freq="D", tz="UTC")
    close = np.linspace(100, 140, len(ts)) + np.sin(np.arange(len(ts)))
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": "XYZ",
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.linspace(1000, 2000, len(ts)),
    })


def test_features_are_causal_and_target_is_future():
    df = _bars()
    feat = add_quant_features(df, windows=(5, 20))
    with_target = add_forward_target(feat, 3)
    assert with_target.loc[0, "target_return"] == with_target.loc[0, "target_return"]
    assert pd.isna(with_target.loc[0, "momentum_5"])
    assert pd.isna(with_target.iloc[-1]["target_return"])


def test_regime_features_exist():
    out = add_regime_features(_bars())
    assert {"trend_strength", "realized_vol", "vol_regime_z", "regime_label"}.issubset(out.columns)
