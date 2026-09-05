from __future__ import annotations

from dataclasses import asdict
import pandas as pd

from trading_intelligence.research.experiments import ExperimentResult


def experiments_to_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results]).sort_values("sharpe", ascending=False).reset_index(drop=True) if results else pd.DataFrame()


def research_summary(results: list[ExperimentResult]) -> dict[str, object]:
    table = experiments_to_frame(results)
    if table.empty:
        return {"best_by_sharpe": None, "experiments": 0}
    best = table.iloc[0].to_dict()
    return {"best_by_sharpe": best, "experiments": int(len(table)), "table": table}
