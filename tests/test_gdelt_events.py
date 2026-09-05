from datetime import date
import io
import zipfile

import pandas as pd

from trading_intelligence.data.acquisition.gdelt_events import EVENT_COLUMNS, extract_event_table, audit_gdelt_event_table, _url


def test_gdelt_schema_is_61_fields():
    assert len(EVENT_COLUMNS) == 61
    assert EVENT_COLUMNS[0] == "GlobalEventID"
    assert EVENT_COLUMNS[-1] == "SOURCEURL"


def test_gdelt_url():
    assert _url(date(2026, 9, 2)).endswith("/20260902.export.CSV.zip")


def test_extract_full_schema(tmp_path):
    row = [""] * 61
    row[0] = "123"
    row[1] = "20260902"
    row[29] = "1"
    row[30] = "2.5"
    row[31] = "4"
    row[32] = "2"
    row[33] = "3"
    row[34] = "1.2"
    row[59] = "20260902153000"
    payload = "\t".join(row) + "\n"
    z = tmp_path / "20260902.export.CSV.zip"
    with zipfile.ZipFile(z, "w") as archive:
        archive.writestr("20260902.export.CSV", payload)
    out = tmp_path / "events.csv"
    extract_event_table(z, out)
    df = pd.read_csv(out)
    assert len(df) == 1
    assert len(df.columns) == 64  # 61 source fields + two normalized timestamps
    assert df.loc[0, "GlobalEventID"] == 123
    assert pd.notna(df.loc[0, "availability_time"])


def test_gdelt_audit_passes():
    df = pd.DataFrame([
        {
            "GlobalEventID": 1,
            "SQLDATE": "20260902",
            "DATEADDED": "20260902153000",
            "event_date": pd.Timestamp("2026-09-02", tz="UTC"),
            "availability_time": pd.Timestamp("2026-09-02 15:30:00", tz="UTC"),
            "EventCode": "010",
            "GoldsteinScale": 2.0,
            "AvgTone": 1.0,
        }
    ])
    assert audit_gdelt_event_table(df)["status"] == "PASS"
