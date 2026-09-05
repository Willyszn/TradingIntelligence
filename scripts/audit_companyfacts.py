from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description='Stream-audit normalized SEC Company Facts.')
    p.add_argument('path', type=Path)
    p.add_argument('--chunksize', type=int, default=250_000)
    a=p.parse_args()
    required=['cik','entity_name','taxonomy','tag','unit','filed','form','val']
    rows=0; chunks=0; nulls={k:0 for k in required}; ciks=set(); tax={}; forms={}; min_filed=None; max_filed=None
    for df in pd.read_csv(a.path, chunksize=a.chunksize, low_memory=False):
        chunks += 1; rows += len(df)
        for k in required:
            if k in df.columns: nulls[k] += int(df[k].isna().sum())
        if 'cik' in df.columns: ciks.update(pd.to_numeric(df['cik'], errors='coerce').dropna().astype(int).unique().tolist())
        if 'taxonomy' in df.columns:
            for k,v in df['taxonomy'].fillna('NA').value_counts().items(): tax[str(k)] = tax.get(str(k),0)+int(v)
        if 'form' in df.columns:
            for k,v in df['form'].fillna('NA').value_counts().items(): forms[str(k)] = forms.get(str(k),0)+int(v)
        if 'filed' in df.columns:
            d=pd.to_datetime(df['filed'], errors='coerce', utc=True).dropna()
            if len(d):
                lo,hi=d.min(),d.max(); min_filed=lo if min_filed is None or lo<min_filed else min_filed; max_filed=hi if max_filed is None or hi>max_filed else max_filed
    print(json.dumps({'rows':rows,'chunks':chunks,'unique_ciks':len(ciks),'required_nulls':nulls,'filed_start':None if min_filed is None else min_filed.isoformat(),'filed_end':None if max_filed is None else max_filed.isoformat(),'taxonomies':tax,'top_forms':sorted(forms.items(), key=lambda x:x[1], reverse=True)[:20]}, indent=2))
if __name__=='__main__': main()
