import pandas as pd

from trading_intelligence.research.cross_sectional import rank_cross_section, top_bottom_portfolio


def test_cross_sectional_ranking_is_per_timestamp():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01"] * 4, utc=True),
        "symbol": ["A", "B", "C", "D"],
        "expected_return": [0.04, 0.02, -0.01, -0.03],
    })
    out = rank_cross_section(frame)
    assert out.loc[out["symbol"] == "A", "cross_sectional_rank"].iloc[0] > 0.9
    assert out.loc[out["symbol"] == "D", "cross_sectional_rank"].iloc[0] <= 0.25


def test_top_bottom_portfolio_is_dollar_balanced_by_group():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01"] * 10, utc=True),
        "symbol": list("ABCDEFGHIJ"),
        "expected_return": list(range(10)),
    })
    out = top_bottom_portfolio(frame, quantile=0.2)
    assert abs(out[out["portfolio_weight"] > 0]["portfolio_weight"].sum() - 1.0) < 1e-9
    assert abs(out[out["portfolio_weight"] < 0]["portfolio_weight"].sum() + 1.0) < 1e-9
