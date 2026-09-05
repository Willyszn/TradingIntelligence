from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import pandas as pd

from trading_intelligence.models.baseline import QuantBaselineModel
from trading_intelligence.research.backtest import BacktestConfig, run_signal_backtest
from trading_intelligence.research.metrics import summarize_strategy_returns
from trading_intelligence.research.splits import chronological_purged_split


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    rows: int
    mean_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    directional_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_experiment(
    frame: pd.DataFrame,
    name: str,
    feature_columns: list[str],
    train_fraction: float = 0.7,
    cost_bps_per_side: float = 5.0,
    holding_bars: int = 3,
    seed: int = 42,
) -> ExperimentResult:
    required = {"timestamp", "symbol", "close", "target_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Experiment frame missing columns: {sorted(missing)}")
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")

    usable = frame.dropna(subset=feature_columns + ["target_return"]).copy()
    if len(usable) < 100:
        raise ValueError("At least 100 usable observations are required")
    train, test = chronological_purged_split(usable, train_fraction=train_fraction)
    if len(train) < 50 or len(test) < 20:
        raise ValueError("Insufficient train/test observations after chronological split and purge")
    model = QuantBaselineModel(random_state=seed)
    model.fit(train, feature_columns)
    test["expected_return"] = model.predict(test)
    bt = run_signal_backtest(
        test,
        config=BacktestConfig(
            cost_bps_per_side=cost_bps_per_side,
            holding_bars=holding_bars,
        ),
    )
    summary = summarize_strategy_returns(bt["net_return"])
    direction = (test["expected_return"] > 0) == (test["target_return"] > 0)
    return ExperimentResult(
        name=name,
        rows=len(bt),
        mean_return=summary.get("mean_return", float("nan")),
        sharpe=summary.get("sharpe", float("nan")),
        max_drawdown=summary.get("max_drawdown", float("nan")),
        win_rate=summary.get("win_rate", float("nan")),
        directional_accuracy=float(direction.mean()),
    )


def run_experiment_suite(
    frame: pd.DataFrame,
    experiment_specs: dict[str, list[str]],
    train_fraction: float = 0.7,
    cost_bps_per_side: float = 5.0,
    holding_bars: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a fixed apples-to-apples model suite over multiple feature sets."""
    results = []
    for name, columns in experiment_specs.items():
        result = run_experiment(
            frame,
            name=name,
            feature_columns=columns,
            train_fraction=train_fraction,
            cost_bps_per_side=cost_bps_per_side,
            holding_bars=holding_bars,
            seed=seed,
        )
        results.append(result.to_dict())
    return pd.DataFrame(results).sort_values(
        ["sharpe", "mean_return"], ascending=False, na_position="last"
    ).reset_index(drop=True)


def save_experiment_report(results: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results.to_json(orient="records", indent=2), encoding="utf-8")
