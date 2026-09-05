from pathlib import Path
import pandas as pd
from trading_intelligence.research.pit_audit import audit_pit_fact_events

def test_pit_audit(tmp_path: Path):
    p = tmp_path / "pit.csv"
    pd.DataFrame([
        {"cik":1,"entity_name":"A","taxonomy":"us-gaap","tag":"Assets","unit":"USD","start":"","end":"2024-12-31","filed":"2025-02-01T00:00:00Z","form":"10-K","accn":"a","val":100,"information_time":"2025-02-01T18:00:00Z","metric":"assets"},
        {"cik":1,"entity_name":"A","taxonomy":"us-gaap","tag":"Assets","unit":"USD","start":"","end":"2024-12-31","filed":"2025-02-01T00:00:00Z","form":"10-K","accn":"a","val":100,"information_time":"2025-02-01T18:00:00Z","metric":"assets"},
    ]).to_csv(p, index=False)
    r = audit_pit_fact_events(p, chunksize=1)
    assert r["rows"] == 2
    assert r["unique_ciks"] == 1
    assert r["duplicate_event_keys"] == 1
    assert r["information_time_parse_failures"] == 0
