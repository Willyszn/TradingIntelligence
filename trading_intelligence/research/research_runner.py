from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd

from trading_intelligence.research.assembly import assemble_research_dataset
from trading_intelligence.research.experiment_runner import SuiteConfig, default_experiment_specs
from trading_intelligence.research.experiments import run_experiment_suite
from trading_intelligence.research.statistics import diagnose_predictions, multiple_testing_warning
from trading_intelligence.models.baseline import QuantBaselineModel
from trading_intelligence.research.splits import chronological_purged_split


@dataclass(frozen=True)
class ResearchRun:
    dataset_rows: int
    symbols: int
    experiments: int
    ranking: list[dict]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def run_research_from_market_dir(market_dir: str | Path, output_json: str | Path, *, config: SuiteConfig | None = None) -> ResearchRun:
    cfg = config or SuiteConfig()
    dataset, assembly = assemble_research_dataset(market_dir, horizon=cfg.holding_bars)
    if not assembly.readiness.ready_for_ml:
        raise ValueError("Dataset is not ready for meaningful ML research; inspect the readiness report first.")

    specs = default_experiment_specs(dataset)
    results = run_experiment_suite(
        dataset,
        specs,
        train_fraction=cfg.train_fraction,
        cost_bps_per_side=cfg.cost_bps_per_side,
        holding_bars=cfg.holding_bars,
        seed=cfg.seed,
    )
    ranking = results.to_dict(orient="records") if isinstance(results, pd.DataFrame) else list(results)
    ranking = sorted(ranking, key=lambda r: float(r.get("sharpe", float("-inf"))), reverse=True)
    warnings: list[str] = []
    warning = multiple_testing_warning(len(ranking))
    if warning:
        warnings.append(warning)

    report = ResearchRun(
        dataset_rows=int(len(dataset)),
        symbols=int(dataset["symbol"].nunique()),
        experiments=len(ranking),
        ranking=ranking,
        warnings=warnings,
    )
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return report
