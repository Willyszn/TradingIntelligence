import numpy as np
import pandas as pd

from trading_intelligence.data.providers.synthetic import SyntheticMarketDataProvider
from trading_intelligence.research.pipeline import prepare_research_frame, run_research_ladder
from trading_intelligence.research.backtest import run_signal_backtest, BacktestConfig


def test_synthetic_pipeline_and_ladder():
    bars = SyntheticMarketDataProvider(seed=7).generate(("AAA", "BBB"), periods=320)
    frame = prepare_research_frame(bars, horizon_bars=3)
    assert "target_return" in frame
    results = run_research_ladder(frame.dropna(subset=["target_return"]), cost_bps_per_side=1.0)
    assert {r.name for r in results} >= {"quant_only", "regime_aware"}
    assert all(r.rows > 0 for r in results)


def test_backtest_deducts_cost_only_when_trading():
    dates = pd.date_range("2024-01-01", periods=5, tz="UTC")
    frame = pd.DataFrame({
        "timestamp": dates, "symbol": ["AAA"] * 5,
        "open": [100, 101, 102, 103, 104],
        "close": [100, 101, 102, 103, 104],
        "expected_return": [0.1, 0.1, 0.1, 0.1, 0.1],
    })
    out = run_signal_backtest(frame, config=BacktestConfig(cost_bps_per_side=5, holding_bars=1))
    assert np.all(out.loc[out["signal"] != 0, "cost"] > 0)
    assert (out["net_return"] <= out["gross_return"]).all()
