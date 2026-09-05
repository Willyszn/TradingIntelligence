import pandas as pd
from trading_intelligence.research.pit_fundamentals import attach_information_time, latest_asof, make_fundamental_snapshot

def test_attach_information_time_prefers_acceptance():
    cf = pd.DataFrame([{'cik':1,'entity_name':'X','taxonomy':'us-gaap','tag':'Assets','unit':'USD','filed':'2024-01-02','form':'10-K','start':None,'end':'2023-12-31','accn':'0001','val':'100'}])
    sub = pd.DataFrame([{'cik':1,'accessionNumber':'0001','acceptanceDateTime':'2024-01-02T15:30:00Z'}])
    out = attach_information_time(cf, sub)
    assert out.loc[0,'information_time'] == pd.Timestamp('2024-01-02T15:30:00Z')

def test_latest_asof_excludes_future_revision():
    facts = pd.DataFrame([
        {'cik':1,'tag':'Assets','unit':'USD','start':None,'end':'2023-12-31','val':'100','accn':'a','information_time':'2024-01-02T10:00:00Z'},
        {'cik':1,'tag':'Assets','unit':'USD','start':None,'end':'2023-12-31','val':'110','accn':'b','information_time':'2024-02-01T10:00:00Z'},
    ])
    out = latest_asof(facts, pd.Timestamp('2024-01-15T00:00:00Z'))
    assert len(out) == 1 and float(out.iloc[0]['val_num']) == 100

def test_snapshot_creates_margin():
    latest = pd.DataFrame([
        {'cik':1,'metric':'revenue','unit':'USD','start':'2023-01-01','end':'2023-12-31','val_num':200,'information_time':'2024-02-01'},
        {'cik':1,'metric':'net_income','unit':'USD','start':'2023-01-01','end':'2023-12-31','val_num':20,'information_time':'2024-02-01'},
    ])
    snap = make_fundamental_snapshot(latest)
    assert abs(float(snap.iloc[0]['net_margin']) - 0.1) < 1e-9
