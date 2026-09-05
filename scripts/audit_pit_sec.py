from __future__ import annotations
import argparse, json
from pathlib import Path
from trading_intelligence.research.pit_audit import audit_pit_fact_events

p = argparse.ArgumentParser(description="Audit the point-in-time SEC fact event dataset.")
p.add_argument("path", type=Path)
p.add_argument("--chunksize", type=int, default=250_000)
args = p.parse_args()
print(json.dumps(audit_pit_fact_events(args.path, chunksize=args.chunksize), indent=2))
