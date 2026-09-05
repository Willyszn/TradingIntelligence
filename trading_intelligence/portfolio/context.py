from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioContext:
    account_equity: float
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    existing_risk_pct: float = 0.0

    def validate(self) -> None:
        if self.account_equity <= 0:
            raise ValueError("account_equity must be positive")
