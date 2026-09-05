from __future__ import annotations

import argparse
from pathlib import Path

from trading_intelligence.research.experiment_runner import SuiteConfig, run_suite_from_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the controlled model comparison suite.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/reports/experiment_suite.json"))
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--cost-bps-per-side", type=float, default=5.0)
    parser.add_argument("--holding-bars", type=int, default=3)
    args = parser.parse_args()
    results = run_suite_from_csv(
        args.input_csv,
        args.output,
        SuiteConfig(
            train_fraction=args.train_fraction,
            cost_bps_per_side=args.cost_bps_per_side,
            holding_bars=args.holding_bars,
        ),
    )
    print(results.to_string(index=False))
    print(f"\nSaved report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
