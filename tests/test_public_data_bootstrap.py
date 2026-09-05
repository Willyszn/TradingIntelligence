from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading_intelligence.data.acquisition.binance_public import download_daily_klines
from trading_intelligence.data.acquisition.gdelt_events import download_event_days, extract_event_table


class FakeResponse:
    def __init__(self, payload=None, content: bytes = b"", status_code: int = 200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_binance_public_klines_normalize(tmp_path: Path):
    rows = [
        [1704067200000, "42000", "43000", "41000", "42500", "10", 1704153599999, "425000", 100, "5", "212500", "0"],
        [1704153600000, "42500", "44000", "42000", "43500", "11", 1704239999999, "478500", 110, "5.5", "239250", "0"],
    ]
    with patch("trading_intelligence.data.acquisition.binance_public.requests.Session.get", return_value=FakeResponse(payload=rows)):
        out = download_daily_klines("BTCUSDT", tmp_path / "btc.csv", start=datetime(2024, 1, 1, tzinfo=timezone.utc))
    df = pd.read_csv(out)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert len(df) == 2
    assert df.loc[0, "symbol"] == "BTCUSDT"
    assert float(df.loc[1, "close"]) == 43500.0


def test_gdelt_download_skips_missing_days_and_extracts(tmp_path: Path):
    with patch("trading_intelligence.data.acquisition.gdelt_events.requests.Session.get", return_value=FakeResponse(status_code=404)):
        paths = download_event_days(date(2026, 1, 1), date(2026, 1, 2), tmp_path / "events")
    assert paths == []

    # Build a valid 61-field GDELT 2.0 event archive fixture.
    import zipfile
    from trading_intelligence.data.acquisition.gdelt_events import EVENT_COLUMNS
    fields = [""] * len(EVENT_COLUMNS)
    fields[0] = "1"
    fields[1] = "20240101"
    fields[5] = "USA"
    fields[6] = "US ACTOR"
    fields[7] = "USA"
    fields[15] = "RUS"
    fields[16] = "RU ACTOR"
    fields[17] = "RUS"
    fields[25] = "1"
    fields[26] = "010"
    fields[27] = "010"
    fields[28] = "01"
    fields[29] = "1"
    fields[30] = "1.5"
    fields[31] = "2"
    fields[32] = "2"
    fields[33] = "1"
    fields[34] = "-0.5"
    fields[59] = "20240101120000"
    fields[60] = "https://example.com"
    archive = tmp_path / "20240101.export.CSV.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("20240101.export.CSV", "\t".join(fields) + "\n")
    out = extract_event_table(archive, tmp_path / "out.csv")
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df.loc[0, "GlobalEventID"] == 1
    assert df.loc[0, "GLOBALEVENTID"] == 1
    assert df.loc[0, "GoldsteinScale"] == 1.5
