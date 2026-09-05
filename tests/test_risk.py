from trading_intelligence.data.schemas import TradeIdea
from trading_intelligence.risk.position_size import calculate_position_size


def test_risk_sizing():
    result = calculate_position_size(
        TradeIdea(symbol="XYZ", direction="LONG", entry=150, stop=145, target=165),
        account_equity=50_000,
        risk_per_trade_pct=1.0,
    )
    assert abs(result.quantity - 100) < 1e-9
    assert abs(result.risk_dollars - 500) < 1e-9
    assert abs(result.rr_ratio - 3.0) < 1e-9


def test_short_directional_validation_and_multiplier():
    result = calculate_position_size(
        TradeIdea(symbol="ES", direction="SHORT", entry=5000, stop=5010, target=4970, contract_multiplier=50),
        account_equity=100_000,
        risk_per_trade_pct=1.0,
        quantity_increment=1,
    )
    assert result.quantity == 2
    assert abs(result.risk_dollars - 1000) < 1e-9
