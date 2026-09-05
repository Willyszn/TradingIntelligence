import zipfile
import pandas as pd
from trading_intelligence.data.acquisition.gdelt_events_pipeline import process_gdelt_archives,audit_processed_gdelt

def row(gid=1,day='20260902'):
    r=['']*61; r[0]=str(gid); r[1]=day; r[26]='010'; r[29]='1'; r[30]='2.5'; r[31]='4'; r[32]='2'; r[33]='3'; r[34]='1.2'; r[59]=day+'153000'; return r

def test_pipeline(tmp_path):
    raw=tmp_path/'raw'; out=tmp_path/'out'; raw.mkdir(); z=raw/'20260902.export.CSV.zip'
    with zipfile.ZipFile(z,'w') as a: a.writestr('x.CSV','\t'.join(row())+'\n')
    r=process_gdelt_archives(raw,out); assert r['files']==1 and r['output_rows']==1 and r['duplicate_ids']==0
    df=pd.read_csv(out/'20260902.events.csv.gz',compression='gzip'); assert int(df.iloc[0]['GlobalEventID'])==1
    assert audit_processed_gdelt(out)['status']=='PASS'

def test_duplicates_warn(tmp_path):
    raw=tmp_path/'raw'; out=tmp_path/'out'; raw.mkdir()
    for d in ['20260902','20260903']:
        z=raw/f'{d}.export.CSV.zip'
        with zipfile.ZipFile(z,'w') as a: a.writestr('x.CSV','\t'.join(row(9,d))+'\n')
    r=process_gdelt_archives(raw,out); assert r['duplicate_ids']==1
