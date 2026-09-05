from __future__ import annotations
import argparse, json
from pathlib import Path
from trading_intelligence.data.acquisition.sec_bulk import import_companyfacts_zip, import_submissions_zip
from trading_intelligence.data.paths import data_root

p = argparse.ArgumentParser(description='Import SEC bulk companyfacts/submissions archives.')
p.add_argument('--companyfacts', type=Path)
p.add_argument('--submissions', type=Path)
p.add_argument('--out', type=Path, default=None)
a = p.parse_args()
if not a.companyfacts and not a.submissions:
    raise SystemExit('Provide --companyfacts and/or --submissions')
out = a.out or (data_root() / 'fundamentals')
results = {}
if a.companyfacts:
    results['companyfacts'] = import_companyfacts_zip(a.companyfacts, out)
if a.submissions:
    results['submissions'] = import_submissions_zip(a.submissions, data_root() / 'events')
print(json.dumps(results, indent=2, default=str))
