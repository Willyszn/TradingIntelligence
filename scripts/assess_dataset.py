from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trading_intelligence.research.readiness import assess_market_dataset


def main() -> int:
    p = argparse.ArgumentParser(description="Assess whether a market dataset is ready for ML research.")
    p.add_argument("input_csv", type=Path)
    p.add_argument("--min-rows-per-symbol", type=int, default=750)
    p.add_argument("--min-symbols", type=int, default=5)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    frame = pd.read_csv(args.input_csv)
    result = assess_market_dataset(frame, min_rows_per_symbol=args.min_rows_per_symbol, min_symbols=args.min_symbols)
    print(json.dumps(result.to_dict(), indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return 0 if result.status != "NOT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
