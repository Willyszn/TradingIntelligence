from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    if r.empty or r.std(ddof=1) == 0:
        return float("nan")
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year))


def summarize_strategy_returns(returns: pd.Series, periods_per_year: float = 252.0) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {"observations": 0.0}
    return {
        "observations": float(len(r)),
        "mean_return": float(r.mean()),
        "volatility": float(r.std(ddof=1)),
        "sharpe": annualized_sharpe(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "win_rate": float((r > 0).mean()),
    }
