from __future__ import annotations
import argparse,json
from trading_intelligence.data.acquisition.gdelt_events_pipeline import audit_processed_gdelt

def main():
    p=argparse.ArgumentParser(); p.add_argument('directory'); a=p.parse_args(); print(json.dumps(audit_processed_gdelt(a.directory),indent=2))
if __name__=='__main__': main()
