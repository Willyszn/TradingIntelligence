from __future__ import annotations
import argparse, os
from pathlib import Path
from trading_intelligence.research.pit_fundamentals_streaming import build_acceptance_index, build_pit_fact_events, load_sec_map

def main() -> int:
    p=argparse.ArgumentParser(description='Build point-in-time SEC fact events using a compact accession index.')
    p.add_argument('--companyfacts', type=Path, required=True)
    p.add_argument('--submissions', type=Path, required=True)
    p.add_argument('--cik-map', type=Path, required=True)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--chunksize', type=int, default=250_000)
    args=p.parse_args()
    root=Path(os.environ.get('TI_DATA_ROOT', Path.home()/'TradingIntelligenceData'))
    out=args.out_dir or (root/'processed'/'pit_sec')
    out.mkdir(parents=True, exist_ok=True)
    ciks=load_sec_map(args.cik_map)
    idx=out/'acceptance_index.sqlite'
    print(build_acceptance_index(args.submissions, idx, ciks=ciks, chunksize=args.chunksize))
    facts=out/'selected_pit_fact_events.csv'
    print(build_pit_fact_events(args.companyfacts, idx, facts, ciks=ciks, chunksize=args.chunksize))
    return 0
if __name__=='__main__': raise SystemExit(main())
