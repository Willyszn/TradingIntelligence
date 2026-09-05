from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_intelligence.data.acquisition.stooq import download_daily as stooq_download
from trading_intelligence.data.acquisition.fred import download_series as fred_download


class FakeResponse:
    def __init__(self, text: str = "", payload: dict | None = None):
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        assert self._payload is not None
        return self._payload


def test_stooq_normalizes_daily_csv(tmp_path: Path):
    text = "date,open,high,low,close,volume\n2024-01-02,10,11,9,10.5,1000\n2024-01-03,10.5,12,10,11.5,1200\n"
    with patch("trading_intelligence.data.acquisition.stooq.requests.get", return_value=FakeResponse(text=text)):
        out = stooq_download("aapl.us", tmp_path / "aapl.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert df["symbol"].eq("aapl.us").all()
    assert len(df) == 2


def test_fred_download_normalizes_observations(tmp_path: Path):
    payload = {"observations": [
        {"date": "2024-01-01", "value": "5.33"},
        {"date": "2024-01-02", "value": "."},
    ]}
    with patch("trading_intelligence.data.acquisition.fred.requests.get", return_value=FakeResponse(payload=payload)):
        out = fred_download("DFF", "key", tmp_path / "dff.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == ["timestamp", "value", "series_id"]
    assert df.loc[0, "value"] == 5.33
    assert pd.isna(df.loc[1, "value"])
    assert df["series_id"].eq("DFF").all()
