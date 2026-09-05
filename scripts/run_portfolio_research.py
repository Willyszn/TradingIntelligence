from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from trading_intelligence.research.dataset import load_market_csv, build_supervised_market_dataset
from trading_intelligence.research.pipeline import candidate_feature_sets
from trading_intelligence.research.experiments import run_experiment_suite
from trading_intelligence.research.model_selection import select_research_candidate
from trading_intelligence.research.portfolio_backtest import run_equal_weight_event_portfolio, PortfolioBacktestConfig


def load_dir(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {path}")
    frames = [load_market_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Run multi-market research and an equal-weight portfolio backtest")
    p.add_argument("--market-dir", type=Path, default=Path("data/raw/market"))
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=Path("data/research/portfolio_results.csv"))
    args = p.parse_args()

    market = load_dir(args.market_dir)
    frame = build_supervised_market_dataset(market, horizon=args.horizon)
    specs = candidate_feature_sets(frame)
    specs = {k: v for k, v in specs.items() if v}
    if not specs:
        raise SystemExit("No usable feature sets found.")

    results = run_experiment_suite(frame, specs, cost_bps_per_side=args.cost_bps, holding_bars=args.horizon)
    decision = select_research_candidate(results)

    # Demonstrate portfolio-level aggregation using the best model's predictions.
    best_name = decision.candidate if decision.candidate else results.iloc[0]["name"]
    best_features = specs[best_name]
    from trading_intelligence.models.baseline import QuantBaselineModel
    from trading_intelligence.research.splits import chronological_purged_split

    usable = frame.dropna(subset=best_features + ["target_return"]).copy()
    train, test = chronological_purged_split(usable, train_fraction=0.7)
    model = QuantBaselineModel(random_state=42)
    model.fit(train, best_features)
    test["expected_return"] = model.predict(test)
    portfolio = run_equal_weight_event_portfolio(
        test,
        config=PortfolioBacktestConfig(cost_bps_per_side=args.cost_bps, holding_bars=args.horizon, max_positions=10),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    portfolio.to_csv(args.output.with_name(args.output.stem + "_portfolio.csv"), index=False)
    print(results.to_string(index=False))
    print("\nPRELIMINARY RESEARCH GATE:")
    print(decision.to_dict())
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
