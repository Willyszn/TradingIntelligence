from pathlib import Path
import sqlite3
import pandas as pd
from trading_intelligence.research.pit_fundamentals_streaming import build_acceptance_index, build_pit_fact_events

def test_streaming_pit_sec(tmp_path: Path):
    subs = tmp_path/'sub.csv'
    subs.write_text('cik,accessionNumber,acceptanceDateTime,filingDate,form\n1,0001-01-01,2024-01-15T18:30:00.000Z,2024-01-15,10-K\n2,0002-01-01,2024-01-20T19:00:00.000Z,2024-01-20,10-Q\n')
    cf=tmp_path/'cf.csv'
    pd.DataFrame([
        {'cik':1,'entity_name':'A','taxonomy':'us-gaap','tag':'Assets','label':'Assets','description':'','unit':'USD','start':None,'end':'2023-12-31','filed':'2024-01-15','form':'10-K','frame':'CY2023','fy':2023,'fp':'FY','accn':'0001-01-01','val':100},
        {'cik':2,'entity_name':'B','taxonomy':'us-gaap','tag':'Assets','label':'Assets','description':'','unit':'USD','start':None,'end':'2023-12-31','filed':'2024-01-20','form':'10-Q','frame':'CY2023','fy':2023,'fp':'Q4','accn':'0002-01-01','val':200},
        {'cik':3,'entity_name':'C','taxonomy':'us-gaap','tag':'Assets','label':'Assets','description':'','unit':'USD','start':None,'end':'2023-12-31','filed':'2024-01-20','form':'10-K','frame':'CY2023','fy':2023,'fp':'FY','accn':'0003-01-01','val':300},
    ]).to_csv(cf,index=False)
    db=tmp_path/'idx.sqlite'; out=tmp_path/'pit.csv'
    r=build_acceptance_index(subs,db,ciks={1,2})
    assert r['rows_indexed']==2
    z=build_pit_fact_events(cf,db,out,ciks={1,2})
    assert z['rows_written']==2
    df=pd.read_csv(out)
    assert df['information_time'].notna().all()
    assert set(df['cik'])=={1,2}
