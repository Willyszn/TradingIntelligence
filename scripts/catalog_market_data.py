from __future__ import annotations
import argparse
from pathlib import Path
from trading_intelligence.data.catalog import catalog_market_directory
from trading_intelligence.data.paths import market_root, reports_root


def main() -> int:
    p = argparse.ArgumentParser(description="Catalog local market datasets.")
    p.add_argument('--root', type=Path, default=market_root())
    p.add_argument('--out', type=Path, default=reports_root() / 'market_catalog.csv')
    args = p.parse_args()
    df = catalog_market_directory(args.root)
    if df.empty:
        print('No readable market CSV files found.')
        return 1
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f'\nSaved catalog -> {out}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
