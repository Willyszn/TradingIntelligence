from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from trading_intelligence.research.assembly import load_market_directory
from trading_intelligence.research.dataset import build_supervised_market_dataset
from trading_intelligence.research.research_report import ResearchRecord, write_records, write_table
from trading_intelligence.research.benchmarks import naive_zero_predictions, sign_accuracy

FEATURES=["ret_1","ret_5","sma_5_dist","sma_20_dist","volatility_20","volume_z_20","atr_14_pct"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--market-dir",type=Path,default=Path("data/raw/market"))
    p.add_argument("--horizon",type=int,default=3)
    p.add_argument("--min-train",type=int,default=500)
    p.add_argument("--out-dir",type=Path,default=Path("data/reports/initial_research"))
    a=p.parse_args()
    market=load_market_directory(a.market_dir)
    ds=build_supervised_market_dataset(market,horizon=a.horizon)
    usable=[c for c in FEATURES if c in ds.columns]
    ds=ds.dropna(subset=usable+["forward_return","entry_price","exit_price"]).sort_values(["timestamp","symbol"]).reset_index(drop=True)
    records=[]
    if len(ds) < a.min_train+1:
        records.append(ResearchRecord("initial","NOT_READY",{"rows":len(ds),"symbols":int(ds.symbol.nunique())},[f"Need at least {a.min_train+1} usable rows."]))
    else:
        split=int(len(ds)*0.7)
        train=ds.iloc[:split]; test=ds.iloc[split:]
        Xtr=train[usable]; ytr=train["forward_return"]; Xte=test[usable]; yte=test["forward_return"]
        model=Ridge(alpha=1.0).fit(Xtr,ytr)
        pred=model.predict(Xte)
        zero=naive_zero_predictions(test.index)
        records += [
            ResearchRecord("naive_zero","OK",{"rows_train":len(train),"rows_test":len(test),"mae":float(mean_absolute_error(yte,zero)),"rmse":float(mean_squared_error(yte,zero)**0.5),"sign_accuracy":sign_accuracy(yte,zero)},[]),
            ResearchRecord("ridge_quant","OK",{"rows_train":len(train),"rows_test":len(test),"mae":float(mean_absolute_error(yte,pred)),"rmse":float(mean_squared_error(yte,pred)**0.5),"sign_accuracy":sign_accuracy(yte,pred)},[]),
        ]
    write_records(records,a.out_dir/"records.json")
    write_table(records,a.out_dir/"summary.csv")
    for r in records: print(r.name,r.status,r.metrics)
    return 0
if __name__=="__main__": raise SystemExit(main())
