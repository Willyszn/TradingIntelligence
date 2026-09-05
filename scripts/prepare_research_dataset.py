from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from trading_intelligence.research.dataset import load_market_csv, build_supervised_market_dataset, join_point_in_time_news


def main() -> int:
    p = argparse.ArgumentParser(description="Assemble a point-in-time research dataset from normalized market/news files.")
    p.add_argument("--market-dir", default="data/raw/market")
    p.add_argument("--news", default=None)
    p.add_argument("--out", default="data/processed/research_dataset.csv")
    p.add_argument("--horizon", type=int, default=3)
    args = p.parse_args()

    market_paths = sorted(Path(args.market_dir).glob("*.csv"))
    if not market_paths:
        raise SystemExit(f"No market CSV files found in {args.market_dir}")
    market = pd.concat([load_market_csv(p) for p in market_paths], ignore_index=True)
    market = market.drop_duplicates(["symbol", "timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    dataset = build_supervised_market_dataset(market, horizon=args.horizon)

    if args.news:
        news = pd.read_csv(args.news)
        dataset = join_point_in_time_news(dataset, news)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    print(f"Wrote {len(dataset)} rows across {dataset['symbol'].nunique()} symbols to {out}")
    print(f"Feature horizon: {args.horizon} bars")
    print(f"Rows with valid targets: {dataset['forward_return'].notna().sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
