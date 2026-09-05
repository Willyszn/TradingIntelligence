from __future__ import annotations

import argparse
import os
from pathlib import Path

from trading_intelligence.data.acquisition.fred import download_series
from trading_intelligence.data.acquisition.stooq import download_daily


def main() -> int:
    p = argparse.ArgumentParser(description="Download free/public starter datasets for research.")
    p.add_argument("--stooq", nargs="*", default=[], help="Stooq symbols, e.g. aapl.us msft.us spy.us")
    p.add_argument("--fred", nargs="*", default=[], help="FRED series IDs, e.g. DFF VIXCLS DGS10")
    p.add_argument("--fred-api-key", default=os.getenv("FRED_API_KEY"))
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()

    out = Path(args.out)
    if not args.stooq and not args.fred:
        p.error("Provide at least one --stooq or --fred series")

    for symbol in args.stooq:
        path = out / "market" / f"stooq_{symbol.replace('.', '_')}_daily.csv"
        print(f"Downloading Stooq {symbol} -> {path}")
        download_daily(symbol, path)

    if args.fred:
        if not args.fred_api_key:
            p.error("FRED downloads require FRED_API_KEY or --fred-api-key")
        for series in args.fred:
            path = out / "macro" / f"fred_{series}.csv"
            print(f"Downloading FRED {series} -> {path}")
            download_series(series, args.fred_api_key, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
