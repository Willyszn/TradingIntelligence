from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def normalize_ticker(value: str) -> str:
    return str(value).strip().upper().split('.')[0]


def main() -> int:
    p = argparse.ArgumentParser(description='Map selected Stooq symbols to SEC CIKs using SEC entity metadata.')
    p.add_argument('--universe', type=Path, required=True)
    p.add_argument('--entity-master', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    universe = pd.read_csv(args.universe)
    entities = pd.read_csv(args.entity_master)
    if 'selected' not in universe.columns:
        raise SystemExit('Universe report must contain selected column')
    if 'symbol' not in universe.columns:
        raise SystemExit('Universe report must contain symbol column')
    selected = universe[universe['selected'] == True].copy()
    selected['ticker_root'] = selected['symbol'].map(normalize_ticker)
    entities['ticker_list'] = entities['tickers'].fillna('').astype(str).map(
        lambda x: [normalize_ticker(t) for t in x.split(';') if t]
    )
    rows = []
    for _, u in selected.iterrows():
        matches = entities[entities['ticker_list'].map(lambda ts, t=u['ticker_root']: t in ts)]
        for _, e in matches.iterrows():
            rows.append({
                'symbol': u['symbol'],
                'ticker_root': u['ticker_root'],
                'cik': int(e['cik']),
                'entity_name': e['entity_name'],
                'exchanges': e.get('exchanges'),
                'sic': e.get('sic'),
                'sicDescription': e.get('sicDescription'),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        counts = out.groupby('symbol')['cik'].nunique().rename('cik_count')
        out = out.merge(counts, on='symbol', how='left')
        out['mapping_status'] = out['cik_count'].map(lambda n: 'unique' if n == 1 else 'ambiguous')
    else:
        out = pd.DataFrame(columns=['symbol','ticker_root','cik','entity_name','exchanges','sic','sicDescription','cik_count','mapping_status'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'Selected universe symbols: {selected["symbol"].nunique()}')
    print(f'Symbols mapped to SEC: {out["symbol"].nunique()}')
    print(f'Unique mappings: {int((out["mapping_status"] == "unique").sum()) if not out.empty else 0}')
    print(f'Ambiguous mappings: {int((out["mapping_status"] == "ambiguous").sum()) if not out.empty else 0}')
    print(f'Unmapped symbols: {selected["symbol"].nunique() - out["symbol"].nunique()}')
    print(f'Saved -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
