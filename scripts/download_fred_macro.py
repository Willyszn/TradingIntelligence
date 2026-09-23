from __future__ import annotations

import argparse
from pathlib import Path

from trading_intelligence.data.paths import data_root
from trading_intelligence.research.fred_macro import (
    DEFAULT_SERIES,
    download_macro_vintages,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download point-in-time FRED/ALFRED macro vintages."
    )

    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-09-02")
    parser.add_argument("--vintage-start", default="2023-01-01")
    parser.add_argument("--vintage-end", default="2026-09-02")

    args = parser.parse_args()

    output = (
        args.output
        or data_root()
        / "macro"
        / "fred_macro_pit_vintages.csv"
    )

    result = download_macro_vintages(
        output,
        series=DEFAULT_SERIES,
        observation_start=args.start,
        observation_end=args.end,
        vintage_start=args.vintage_start,
        vintage_end=args.vintage_end,
    )

    print(result)


if __name__ == "__main__":
    main()
