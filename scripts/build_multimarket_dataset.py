from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_intelligence.research.assembly import assemble_research_dataset


def main() -> int:
    p = argparse.ArgumentParser(description="Assemble all local market files into a supervised multi-market research dataset.")
    p.add_argument("--market-dir", type=Path, default=Path("data/raw/market"))
    p.add_argument("--news", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("data/processed/research_dataset.csv"))
    p.add_argument("--report", type=Path, default=Path("data/processed/assembly_report.json"))
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--news-lookback-hours", type=int, default=24)
    args = p.parse_args()

    dataset, result = assemble_research_dataset(
        args.market_dir,
        news_csv=args.news,
        horizon=args.horizon,
        news_lookback_hours=args.news_lookback_hours,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "raw_rows": result.raw_rows,
        "processed_rows": result.processed_rows,
        "symbols": result.symbols,
        "market_files": result.market_files,
        "news_rows": result.news_rows,
        "readiness": result.readiness.to_dict(),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {len(dataset)} rows across {dataset['symbol'].nunique()} symbols to {args.out}")
    print(f"Readiness: {result.readiness.status}")
    for warning in result.readiness.warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
