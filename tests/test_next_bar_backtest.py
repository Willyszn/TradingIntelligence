import numpy as np
import pandas as pd

from trading_intelligence.research.backtest import BacktestConfig, run_signal_backtest


def test_backtest_uses_next_bar_open_entry():
    t = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": t,
        "symbol": ["A"] * 5,
        "open": [10, 11, 12, 13, 14],
        "close": [10, 11, 12, 13, 14],
        "expected_return": [1, 0, 0, 0, 0],
    })
    out = run_signal_backtest(df, config=BacktestConfig(holding_bars=3, cost_bps_per_side=0))
    first = out.iloc[0]
    assert first["entry_price"] == 11
    assert first["exit_price"] == 13
    assert np.isclose(first["future_return"], (13 / 11) - 1)
    assert np.isclose(first["gross_return"], first["future_return"])
