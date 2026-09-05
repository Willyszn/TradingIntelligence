import pandas as pd
from trading_intelligence.research.portfolio_backtest import run_equal_weight_event_portfolio, PortfolioBacktestConfig


def test_portfolio_backtest_aggregates_by_timestamp():
    rows = []
    dates = pd.date_range("2025-01-01", periods=6, tz="UTC")
    for sym, base in [("A", 100.0), ("B", 200.0)]:
        for i, ts in enumerate(dates):
            px = base + i
            rows.append({"symbol": sym, "timestamp": ts, "open": px, "close": px + 0.5, "expected_return": 0.1})
    frame = pd.DataFrame(rows)
    out = run_equal_weight_event_portfolio(frame, config=PortfolioBacktestConfig(holding_bars=3))
    assert len(out) == 3
    assert out["positions"].max() == 2
    assert out["portfolio_return"].notna().all()
