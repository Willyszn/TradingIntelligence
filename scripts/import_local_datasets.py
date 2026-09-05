from __future__ import annotations
import argparse, json
from pathlib import Path
from trading_intelligence.data.acquisition.stooq_archive import import_stooq_archive
from trading_intelligence.data.acquisition.sec_bulk import import_companyfacts_zip, import_submissions_zip
from trading_intelligence.data.paths import data_root, market_root

p = argparse.ArgumentParser(description='Import the downloaded Stooq and SEC archives into persistent research storage.')
p.add_argument('--stooq', type=Path)
p.add_argument('--companyfacts', type=Path)
p.add_argument('--submissions', type=Path)
p.add_argument('--data-root', type=Path, default=None)
a = p.parse_args()
if not any([a.stooq, a.companyfacts, a.submissions]):
    raise SystemExit('Provide at least one of --stooq, --companyfacts, --submissions')
root = a.data_root or data_root()
results = {}
if a.stooq:
    results['stooq'] = import_stooq_archive(a.stooq, root / 'market')
if a.companyfacts:
    results['companyfacts'] = import_companyfacts_zip(a.companyfacts, root / 'fundamentals')
if a.submissions:
    results['submissions'] = import_submissions_zip(a.submissions, root / 'events')
(root / 'reports').mkdir(parents=True, exist_ok=True)
(root / 'reports' / 'local_dataset_import.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
print(json.dumps(results, indent=2, default=str))
