from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
from trading_intelligence.research.pit_fundamentals import attach_information_time, latest_asof, make_fundamental_snapshot


def main() -> int:
    p = argparse.ArgumentParser(description='Build point-in-time SEC fundamental snapshots.')
    p.add_argument('--companyfacts', type=Path, required=True)
    p.add_argument('--submissions', type=Path, required=True)
    p.add_argument('--asof', required=True, help='UTC timestamp, e.g. 2026-09-01T23:59:59Z')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    root = Path(os.environ.get('TI_DATA_ROOT', Path.home() / 'TradingIntelligenceData'))
    out = args.out or (root / 'processed' / 'pit_fundamentals')
    out.mkdir(parents=True, exist_ok=True)
    facts = pd.read_csv(args.companyfacts, low_memory=False)
    subs = pd.read_csv(args.submissions, low_memory=False)
    joined = attach_information_time(facts, subs)
    latest = latest_asof(joined, pd.Timestamp(args.asof))
    snap = make_fundamental_snapshot(latest)
    latest_path = out / 'latest_disclosed_facts_asof.csv'
    snap_path = out / 'fundamental_snapshot_asof.csv'
    latest.to_csv(latest_path, index=False)
    snap.to_csv(snap_path, index=False)
    print({'asof': args.asof, 'facts_rows': len(facts), 'latest_rows': len(latest), 'ciks': int(latest['cik'].nunique()) if not latest.empty else 0, 'snapshot_rows': len(snap), 'latest_output': str(latest_path), 'snapshot_output': str(snap_path)})
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
