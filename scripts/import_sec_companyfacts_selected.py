from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from trading_intelligence.data.acquisition.sec_bulk import import_companyfacts_zip


def main() -> int:
    p = argparse.ArgumentParser(description='Import Company Facts only for CIKs mapped to the research universe.')
    p.add_argument('--companyfacts', type=Path, required=True)
    p.add_argument('--cik-map', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    mapping = pd.read_csv(args.cik_map)
    if 'cik' not in mapping.columns:
        raise SystemExit('CIK map must contain cik column')
    ciks = {int(x) for x in mapping['cik'].dropna().unique()}
    result = import_companyfacts_zip(args.companyfacts, args.out, cik_filter=ciks)
    print(result)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
