from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from trading_intelligence.research.pit_fundamental_observations import write_observations


def main() -> int:
    p = argparse.ArgumentParser(description='Build canonical point-in-time SEC fundamental observations.')
    p.add_argument('input_csv', type=Path, help='Streaming PIT SEC fact-events CSV')
    p.add_argument('--out', type=Path, help='Output canonical observation CSV')
    args = p.parse_args()
    root = Path(os.environ.get('TI_DATA_ROOT', Path.home() / 'TradingIntelligenceData'))
    out = args.out or (root / 'processed' / 'pit_fundamentals' / 'canonical_observations.csv')
    stats = write_observations(args.input_csv, out)
    print(json.dumps(stats, indent=2, default=str))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
