import numpy as np
import pandas as pd

from trading_intelligence.research.dataset import build_supervised_market_dataset
from trading_intelligence.research.walk_forward import walk_forward_regression
from trading_intelligence.research.multimarket_runner import pooled_time_split


def _market(n=80, symbols=("AAA", "BBB")):
    rng=np.random.default_rng(1)
    rows=[]
    dates=pd.date_range('2025-01-01', periods=n, freq='D', tz='UTC')
    for sym in symbols:
        close=100+np.cumsum(rng.normal(0,1,n))
        for i,d in enumerate(dates):
            rows.append(dict(timestamp=d,symbol=sym,open=close[i],high=close[i]+1,low=close[i]-1,close=close[i],volume=1000+i))
    return pd.DataFrame(rows)


def test_dataset_has_target_timestamp():
    out=build_supervised_market_dataset(_market(), horizon=3)
    assert 'target_timestamp' in out.columns
    assert out.loc[out.index[3],'target_timestamp'] > out.loc[out.index[3],'timestamp']


def test_pooled_split_keeps_timestamp_boundary_together():
    df=build_supervised_market_dataset(_market(30), horizon=3).dropna(subset=['forward_return']).rename(columns={'forward_return':'target_return'})
    train,test=pooled_time_split(df, 0.7)
    assert train['timestamp'].max() < test['timestamp'].min()
    assert set(train['timestamp']).isdisjoint(set(test['timestamp']))


def test_walk_forward_runs_with_purge():
    df=build_supervised_market_dataset(_market(120, ('AAA',)), horizon=3)
    df=df.assign(x=df['ret_1']).dropna(subset=['target_timestamp','forward_return','x']).rename(columns={'forward_return':'target_return'})
    results=walk_forward_regression(df,['x'],train_size=40,test_size=15,step=15)
    assert isinstance(results,list)
    assert all(r.train_end < r.test_start for r in results)
