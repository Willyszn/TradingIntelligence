from __future__ import annotations

import argparse
import os
from pathlib import Path

from trading_intelligence.data.acquisition.alpha_vantage import download_daily
from trading_intelligence.data.acquisition.gdelt import download_news
from trading_intelligence.data.acquisition.sec import download_submissions


def main() -> int:
    p = argparse.ArgumentParser(description="Download starter research data from free/public sources.")
    p.add_argument("--alpha-vantage-key", default=os.getenv("ALPHAVANTAGE_API_KEY"))
    p.add_argument("--symbols", nargs="+", default=["IBM", "AAPL", "MSFT", "NVDA"])
    p.add_argument("--gdelt-queries", nargs="+", default=["IBM OR AAPL OR MSFT OR NVDA"])
    p.add_argument("--sec-cik", nargs="+", default=["0000051143"])
    p.add_argument("--out", default="data/raw")
    p.add_argument("--outputsize", choices=["compact", "full"], default="compact", help="Alpha Vantage daily history size; free keys should use compact")
    p.add_argument("--skip-gdelt", action="store_true", help="Skip GDELT news download")
    p.add_argument("--skip-sec", action="store_true", help="Skip SEC filings download")
    args = p.parse_args()

    out = Path(args.out)
    if not args.alpha_vantage_key:
        print("SKIP Alpha Vantage: set ALPHAVANTAGE_API_KEY or pass --alpha-vantage-key")
    else:
        for symbol in args.symbols:
            path = out / "market" / f"{symbol}_daily.csv"
            print(f"Downloading {symbol} -> {path}")
            download_daily(symbol, args.alpha_vantage_key, path, outputsize=args.outputsize)

    if not args.skip_gdelt:
        for q in args.gdelt_queries:
            safe = "_".join(q.replace("OR", "").split())[:60]
            path = out / "news" / f"gdelt_{safe}.csv"
            print(f"Downloading GDELT {q!r} -> {path}")
            download_news(q, path)

    if not args.skip_sec:
        for cik in args.sec_cik:
            path = out / "sec" / f"CIK{str(cik).zfill(10)}_submissions.json"
            print(f"Downloading SEC CIK {cik} -> {path}")
            download_submissions(cik, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
