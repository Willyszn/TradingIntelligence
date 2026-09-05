from __future__ import annotations
import argparse, json
from trading_intelligence.data.acquisition.stooq_archive import import_stooq_archive

p = argparse.ArgumentParser(description='Import a Stooq daily ASCII archive into persistent normalized market CSVs.')
p.add_argument('archive')
p.add_argument('--out', default=None)
a = p.parse_args()
from trading_intelligence.data.paths import market_root
out = a.out or market_root()
result = import_stooq_archive(a.archive, out)
print(json.dumps(result, indent=2))
