from __future__ import annotations

import argparse
from trading_intelligence.research.research_runner import run_research_from_market_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled multi-model research suite on local market data.")
    parser.add_argument("--market-dir", default="data/raw/market")
    parser.add_argument("--output", default="data/processed/research_run.json")
    args = parser.parse_args()
    report = run_research_from_market_dir(args.market_dir, args.output)
    print(f"Dataset rows: {report.dataset_rows}")
    print(f"Symbols: {report.symbols}")
    print(f"Experiments: {report.experiments}")
    if report.warnings:
        for warning in report.warnings:
            print(f"WARNING: {warning}")
    if report.ranking:
        print("Top experiment:", report.ranking[0].get("name", "unknown"))


if __name__ == "__main__":
    main()
