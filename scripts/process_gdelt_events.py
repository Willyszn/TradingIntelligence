from __future__ import annotations
import argparse,json
from pathlib import Path
from trading_intelligence.data.paths import data_root
from trading_intelligence.data.acquisition.gdelt_events_pipeline import process_gdelt_archives

def main():
    p=argparse.ArgumentParser(description='Process downloaded GDELT event archives into compact research partitions.')
    p.add_argument('--raw'); p.add_argument('--output'); p.add_argument('--chunksize',type=int,default=100000); a=p.parse_args()
    root=data_root(); raw=Path(a.raw) if a.raw else root/'news'/'gdelt_events'/'raw'; out=Path(a.output) if a.output else root/'news'/'gdelt_events'/'processed'
    print(json.dumps(process_gdelt_archives(raw,out,chunksize=a.chunksize),indent=2))
if __name__=='__main__': main()
