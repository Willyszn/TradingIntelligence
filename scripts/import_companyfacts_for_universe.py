from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from trading_intelligence.data.acquisition.sec_bulk import import_companyfacts_zip


def main() -> int:
    p = argparse.ArgumentParser(description="Import SEC Company Facts only for CIKs in a research-universe mapping.")
    p.add_argument('--companyfacts', type=Path, required=True)
    p.add_argument('--cik-map', type=Path, required=True)
    p.add_argument('--out', type=Path, required=False)
    args = p.parse_args()
    mapping = pd.read_csv(args.cik_map, low_memory=False)
    if 'cik' not in mapping.columns:
        raise SystemExit('CIK map must contain cik column')
    ciks = {int(x) for x in mapping['cik'].dropna().unique()}
    out = args.out or (Path(__import__('os').environ.get('TI_DATA_ROOT', str(Path.home()/'TradingIntelligenceData'))) / 'fundamentals')
    result = import_companyfacts_zip(args.companyfacts, out, cik_filter=ciks)
    print(result)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
