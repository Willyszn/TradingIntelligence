from __future__ import annotations

import math

from trading_intelligence.data.schemas import PositionSizeResult, TradeIdea


def calculate_position_size(
    trade: TradeIdea,
    account_equity: float,
    risk_per_trade_pct: float,
    max_position_value_pct: float | None = None,
    quantity_increment: float | None = None,
    min_quantity: float | None = None,
) -> PositionSizeResult:
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")
    if risk_per_trade_pct <= 0 or risk_per_trade_pct >= 100:
        raise ValueError("risk_per_trade_pct must be between 0 and 100")
    if max_position_value_pct is not None and not (0 < max_position_value_pct <= 100):
        raise ValueError("max_position_value_pct must be between 0 and 100")

    warnings: list[str] = []
    risk_per_unit = abs(trade.entry - trade.stop) * trade.contract_multiplier
    risk_dollars = account_equity * risk_per_trade_pct / 100.0
    quantity = risk_dollars / risk_per_unit
    capital_required = quantity * trade.entry * trade.contract_multiplier

    if max_position_value_pct is not None:
        cap = account_equity * max_position_value_pct / 100.0
        if capital_required > cap:
            quantity = cap / (trade.entry * trade.contract_multiplier)
            capital_required = quantity * trade.entry * trade.contract_multiplier
            risk_dollars = quantity * risk_per_unit
            warnings.append("Position capped by maximum position-value limit.")

    if quantity_increment is not None:
        if quantity_increment <= 0:
            raise ValueError("quantity_increment must be positive")
        quantity = math.floor(quantity / quantity_increment) * quantity_increment
        capital_required = quantity * trade.entry * trade.contract_multiplier
        risk_dollars = quantity * risk_per_unit

    if min_quantity is not None and quantity < min_quantity:
        warnings.append("Calculated quantity is below broker/instrument minimum.")

    if quantity <= 0:
        warnings.append("No feasible quantity after sizing constraints.")

    reward_dollars = None
    rr_ratio = None
    if trade.target is not None:
        reward_per_unit = abs(trade.target - trade.entry) * trade.contract_multiplier
        reward_dollars = quantity * reward_per_unit
        rr_ratio = reward_per_unit / risk_per_unit

    return PositionSizeResult(
        symbol=trade.symbol,
        direction=trade.direction,
        quantity=quantity,
        capital_required=capital_required,
        risk_dollars=risk_dollars,
        risk_pct=risk_dollars / account_equity * 100.0,
        reward_dollars=reward_dollars,
        rr_ratio=rr_ratio,
        warnings=warnings,
    )
