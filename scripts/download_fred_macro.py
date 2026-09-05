from __future__ import annotations

import argparse
from pathlib import Path

from trading_intelligence.research.fred_macro import DEFAULT_SERIES, download_macro_vintages
from trading_intelligence.data.paths import data_root


def main() -> None:
    p = argparse.ArgumentParser(description="Download point-in-time FRED/ALFRED macro vintages.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--start", default="1900-01-01")
    p.add_argument("--end", default="9999-12-31")
    args = p.parse_args()
    output = args.output or (data_root() / "macro" / "fred_macro_vintages.csv")
    print(download_macro_vintages(output, series=DEFAULT_SERIES, observation_start=args.start, observation_end=args.end))


if __name__ == "__main__":
    main()
