from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import pandas as pd


@dataclass(frozen=True)
class PromotionDecision:
    candidate: str
    eligible: bool
    reason: str
    score: float
    sharpe: float
    mean_return: float
    observations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_research_candidate(
    results: pd.DataFrame,
    *,
    min_observations: int = 100,
    min_sharpe: float = 0.25,
    require_positive_mean: bool = True,
) -> PromotionDecision:
    """Conservative research gate; never equates a backtest winner with production readiness."""
    required = {"name", "rows", "sharpe", "mean_return"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Selection results missing columns: {sorted(missing)}")
    if results.empty:
        return PromotionDecision("", False, "No experiment results", float("nan"), float("nan"), float("nan"), 0)

    ranked = results.copy()
    ranked["rows"] = pd.to_numeric(ranked["rows"], errors="coerce")
    ranked["sharpe"] = pd.to_numeric(ranked["sharpe"], errors="coerce")
    ranked["mean_return"] = pd.to_numeric(ranked["mean_return"], errors="coerce")
    ranked["selection_score"] = ranked["sharpe"].fillna(-math.inf)
    eligible_mask = ranked["rows"].fillna(0) >= min_observations
    eligible_mask &= ranked["sharpe"].notna() & (ranked["sharpe"] >= min_sharpe)
    if require_positive_mean:
        eligible_mask &= ranked["mean_return"].notna() & (ranked["mean_return"] > 0)
    eligible_ranked = ranked[eligible_mask].sort_values(["selection_score", "mean_return"], ascending=False)
    row = eligible_ranked.iloc[0] if not eligible_ranked.empty else ranked.sort_values(["selection_score", "mean_return"], ascending=False).iloc[0]
    name = str(row["name"])
    rows = int(row["rows"]) if pd.notna(row["rows"]) else 0
    sharpe = float(row["sharpe"]) if pd.notna(row["sharpe"]) else float("nan")
    mean_return = float(row["mean_return"]) if pd.notna(row["mean_return"]) else float("nan")

    reasons: list[str] = []
    eligible = True
    if rows < min_observations:
        eligible = False
        reasons.append(f"only {rows} observations; need at least {min_observations}")
    if not pd.notna(sharpe) or sharpe < min_sharpe:
        eligible = False
        reasons.append(f"Sharpe {sharpe:.3f} below minimum {min_sharpe:.3f}" if pd.notna(sharpe) else "Sharpe is unavailable")
    if require_positive_mean and (not pd.notna(mean_return) or mean_return <= 0):
        eligible = False
        reasons.append("mean return is not positive")
    if eligible:
        reasons.append("passes preliminary research gate; not a production approval")
    return PromotionDecision(name, eligible, "; ".join(reasons), float(row["selection_score"]), sharpe, mean_return, rows)
