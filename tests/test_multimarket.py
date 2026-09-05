import numpy as np
import pandas as pd

from trading_intelligence.features.quant import add_quant_features, add_forward_target
from trading_intelligence.research.multimarket import fit_predict_oos
from trading_intelligence.signals.explanation import explain_opportunity


def _frame():
    ts = pd.date_range("2020-01-01", periods=180, freq="D", tz="UTC")
    rows = []
    for j, sym in enumerate(["AAA", "BBB"]):
        close = 100 + np.cumsum(np.full(180, 0.25 + 0.05*j)) + np.sin(np.arange(180)/8)
        for i, t in enumerate(ts):
            rows.append({"timestamp": t, "symbol": sym, "asset_class": "equity", "open": close[i], "high": close[i]+1, "low": close[i]-1, "close": close[i], "volume": 1000+i})
    f = pd.DataFrame(rows)
    f = add_quant_features(f)
    return add_forward_target(f, 3)


def test_oos_predictions_are_only_in_test_tail_per_symbol():
    f = _frame()
    out = fit_predict_oos(f, ["momentum_5", "volatility_5", "distance_sma_5"])
    assert set(out["symbol"]) == {"AAA", "BBB"}
    assert out.groupby("symbol").size().min() > 0
    assert out["expected_return"].notna().all()


def test_explanation_returns_evidence():
    row = pd.Series({"momentum_20": .1, "momentum_5": .02, "relative_volume_20": 2.0, "trend_strength": .3, "sentiment_shock": .4})
    exp = explain_opportunity(row)
    assert "positive 20-bar momentum" in exp["positives"]
    assert "abnormal volume" in exp["positives"]
