from pathlib import Path

import pandas as pd

from trading_intelligence.data.acquisition.binance_public import download_daily_klines

def test_binance_downloader_normalizes_klines(monkeypatch, tmp_path: Path):
    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return [[
                1704067200000, "42000", "43000", "41000", "42500", "10",
                1704153599999, "425000", 100, "5", "212500", "0"
            ], [
                1704153600000, "42500", "44000", "42000", "43500", "11",
                1704239999999, "478500", 110, "5.5", "239250", "0"
            ]]

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    out = tmp_path / "btc.csv"
    result = download_daily_klines(
        "BTCUSDT", out,
        start=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
        end=pd.Timestamp("2024-01-03", tz="UTC").to_pydatetime(),
        session=Session(),
    )
    df = pd.read_csv(result)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert len(df) == 2
    assert df.loc[0, "symbol"] == "BTCUSDT"
    assert float(df.loc[0, "close"]) == 42500.0
