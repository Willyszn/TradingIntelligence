import pandas as pd
from trading_intelligence.research.fred_macro import audit_macro_vintages, DEFAULT_SERIES


def test_default_macro_set_is_present_and_named():
    assert len(DEFAULT_SERIES) == 9
    assert DEFAULT_SERIES["EFFR"] == "effective_federal_funds_rate"
    assert DEFAULT_SERIES["VIXCLS"] == "vix"


def test_macro_audit_passes_clean_dataset():
    df = pd.DataFrame([
        {"series_id":"EFFR","observation_date":"2024-01-01","value":5.3,"realtime_start":"2024-01-02","realtime_end":"2262-04-11"},
        {"series_id":"DGS10","observation_date":"2024-01-02","value":4.1,"realtime_start":"2024-01-03","realtime_end":"2262-04-11"},
    ])
    result = audit_macro_vintages(df)
    assert result["status"] == "PASS"
    assert result["duplicate_rows"] == 0


def test_macro_audit_detects_inverted_realtime_window():
    df = pd.DataFrame([
        {"series_id":"EFFR","observation_date":"2024-01-01","value":5.3,"realtime_start":"2024-02-02","realtime_end":"2024-02-01"},
    ])
    result = audit_macro_vintages(df)
    assert result["status"] == "FAIL"
    assert result["realtime_inversion_rows"] == 1
