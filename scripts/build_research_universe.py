from __future__ import annotations

import argparse
from pathlib import Path
from trading_intelligence.data.paths import market_root, reports_root
from trading_intelligence.research.universe import build_liquid_candidate_universe


def main() -> int:
    p = argparse.ArgumentParser(description="Build a configurable liquidity-screened US market research universe.")
    p.add_argument("--root", type=Path, default=market_root())
    p.add_argument("--out", type=Path, default=reports_root() / "research_universe_candidates.csv")
    p.add_argument("--min-history", type=int, default=750)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-median-dollar-volume", type=float, default=20_000_000.0)
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--max-symbols", type=int, default=500)
    p.add_argument("--include-obvious-non-common", action="store_true")
    p.add_argument("--profile", choices=["broad", "core_equity_etf"], default="core_equity_etf", help="Conservative research profile; use broad to retain leveraged/inverse candidates.")
    args = p.parse_args()

    df = build_liquid_candidate_universe(
        args.root,
        min_history_rows=args.min_history,
        min_median_close=args.min_price,
        min_median_dollar_volume=args.min_median_dollar_volume,
        lookback=args.lookback,
        exclude_obvious_non_common=not args.include_obvious_non_common,
        max_symbols=args.max_symbols,
        profile=args.profile,
    )
    if df.empty:
        print(f"No readable market CSVs found under {args.root}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Scanned files: {len(df)}")
    print(f"Eligible before cap: {int(df['eligible'].sum())}")
    print(f"Selected universe size: {int(df['selected'].sum())}")
    print(df[df["selected"]][["symbol", "rows", "first_timestamp", "last_timestamp", "median_dollar_volume_lookback"]].head(25).to_string(index=False))
    print(f"\nSaved universe report -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
