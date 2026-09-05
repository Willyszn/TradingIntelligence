import pandas as pd

from trading_intelligence.research.splits import chronological_purged_split
from trading_intelligence.data.catalog import catalog_market_directory


def test_chronological_split_purges_overlapping_targets():
    ts = pd.date_range('2026-01-01', periods=20, freq='D', tz='UTC')
    df = pd.DataFrame({
        'timestamp': ts,
        'symbol': ['A'] * len(ts),
        'target_return': [0.01] * len(ts),
        'target_timestamp': list(ts[3:]) + [pd.NaT] * 3,
        'x': range(len(ts)),
    })
    train, test = chronological_purged_split(df, train_fraction=0.7)
    assert train['timestamp'].max() < test['timestamp'].min()
    assert train['target_timestamp'].dropna().max() < test['timestamp'].min()


def test_catalog_market_directory(tmp_path):
    root = tmp_path / 'market'
    root.mkdir()
    pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=3, tz='UTC'),
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
    }).to_csv(root / 'aapl.csv', index=False)
    out = catalog_market_directory(root)
    assert len(out) == 1
    assert out.iloc[0]['rows'] == 3
    assert out.iloc[0]['symbols'] == 1
