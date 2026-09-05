from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from trading_intelligence.data.paths import market_root, reports_root
from trading_intelligence.research.assembly import load_market_directory
from trading_intelligence.research.dataset import build_supervised_market_dataset
from trading_intelligence.research.pipeline import candidate_feature_sets
from trading_intelligence.research.experiments import run_experiment_suite, save_experiment_report
from trading_intelligence.research.model_selection import select_research_candidate
from trading_intelligence.research.readiness import assess_market_dataset


def main() -> int:
    p = argparse.ArgumentParser(description="Run the first quantitative research pass on persistent crypto market data.")
    p.add_argument("--market-dir", type=Path, default=market_root())
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=reports_root() / "crypto_quant_research.csv")
    args = p.parse_args()

    market = load_market_directory(args.market_dir)
    crypto_symbols = sorted(str(s) for s in market["symbol"].dropna().unique())
    if not crypto_symbols:
        raise SystemExit("No symbols found.")

    readiness = assess_market_dataset(market)
    print("DATASET READINESS:")
    print(readiness.to_dict())

    frame = build_supervised_market_dataset(market, horizon=args.horizon)
    specs = candidate_feature_sets(frame)
    # Sentiment experiments are skipped until real point-in-time sentiment features exist.
    specs = {k: v for k, v in specs.items() if k in {"quant_only", "regime_aware"} and v}
    if not specs:
        raise SystemExit("No quantitative feature sets are available.")

    usable = frame.dropna(subset=sorted(set(sum(specs.values(), []))) + ["target_return"])
    if len(usable) < 1000:
        raise SystemExit(f"Not research-ready: only {len(usable)} usable supervised rows.")

    results = run_experiment_suite(
        frame,
        specs,
        train_fraction=0.7,
        cost_bps_per_side=args.cost_bps,
        holding_bars=args.horizon,
    )
    decision = select_research_candidate(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    save_experiment_report(results, args.output.with_suffix('.json'))

    meta = {
        "symbols": crypto_symbols,
        "rows": int(len(market)),
        "supervised_rows": int(len(frame)),
        "usable_rows": int(len(usable)),
        "horizon": args.horizon,
        "decision": decision.to_dict(),
    }
    args.output.with_name(args.output.stem + "_meta.json").write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')

    print("\nEXPERIMENT RESULTS:")
    print(results.to_string(index=False))
    print("\nRESEARCH GATE:")
    print(decision.to_dict())
    print(f"\nSaved: {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
