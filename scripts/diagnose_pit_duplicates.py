from __future__ import annotations

import argparse
import json

from trading_intelligence.research.pit_duplicate_diagnostics import diagnose_pit_duplicates


def main() -> None:
    p = argparse.ArgumentParser(description='Diagnose PIT SEC duplicate rows and metric alias collisions.')
    p.add_argument('path')
    p.add_argument('--chunksize', type=int, default=250_000)
    p.add_argument('--sample-limit', type=int, default=25)
    args = p.parse_args()
    result = diagnose_pit_duplicates(args.path, chunksize=args.chunksize, sample_limit=args.sample_limit)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
