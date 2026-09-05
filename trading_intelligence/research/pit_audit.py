from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = ["cik","entity_name","taxonomy","tag","unit","start","end","filed","form","accn","val","information_time"]


def audit_pit_fact_events(path: str | Path, *, chunksize: int = 250_000) -> dict:
    path = Path(path)
    rows = 0
    chunks = 0
    unique_ciks: set[int] = set()
    nulls = {c: 0 for c in REQUIRED}
    bad_time = 0
    fallback_like = 0
    info_after_filed = 0
    duplicate_keys = 0
    exact_duplicate_keys = 0
    economic_duplicate_keys = 0
    seen_exact: set[tuple] = set()
    seen_economic: set[tuple] = set()
    min_info = None
    max_info = None
    metrics: dict[str, int] = {}

    usecols = None
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=lambda c: c in REQUIRED + ["metric"]):
        chunks += 1
        rows += len(chunk)
        for c in REQUIRED:
            if c in chunk:
                nulls[c] += int(chunk[c].isna().sum())
        if "cik" in chunk:
            unique_ciks.update(int(x) for x in pd.to_numeric(chunk["cik"], errors="coerce").dropna().unique())
        if "metric" in chunk:
            counts = chunk["metric"].fillna("<missing>").value_counts()
            for k, v in counts.items(): metrics[k] = metrics.get(k, 0) + int(v)
        info = pd.to_datetime(chunk["information_time"], utc=True, errors="coerce")
        filed = pd.to_datetime(chunk["filed"], utc=True, errors="coerce")
        bad_time += int(info.isna().sum())
        info_after_filed += int((info.notna() & filed.notna() & (info > filed + pd.Timedelta(days=1))).sum())
        if info.notna().any():
            mi, ma = info.min(), info.max()
            min_info = mi if min_info is None else min(min_info, mi)
            max_info = ma if max_info is None else max(max_info, ma)
        # Distinguish exact filing-event duplicates from economically identical
        # observations that may legitimately occur across filings/restatements.
        econ_cols = [c for c in ["cik","metric","start","end","unit","information_time"] if c in chunk]
        exact_cols = [c for c in ["cik","metric","start","end","unit","information_time","accn","form","filed","val"] if c in chunk]
        econ_keys = chunk[econ_cols].astype("string")
        exact_keys = chunk[exact_cols].astype("string")
        for row in econ_keys.itertuples(index=False, name=None):
            key = tuple(row)
            if key in seen_economic:
                economic_duplicate_keys += 1
            else:
                seen_economic.add(key)
        for row in exact_keys.itertuples(index=False, name=None):
            key = tuple(row)
            if key in seen_exact:
                exact_duplicate_keys += 1
            else:
                seen_exact.add(key)

    duplicate_keys = economic_duplicate_keys

    status = "PASS"
    warnings = []
    if rows == 0:
        status = "FAIL"; warnings.append("No PIT fact rows found.")
    if nulls["information_time"]:
        status = "FAIL"; warnings.append(f"{nulls['information_time']} rows have no information_time.")
    if duplicate_keys:
        warnings.append(f"{duplicate_keys} economic duplicate event keys detected; these may be legitimate repeated/restated facts across filings and require precedence rules during feature aggregation.")
        status = "PASS_WITH_WARNINGS"
    if exact_duplicate_keys:
        warnings.append(f"{exact_duplicate_keys} exact duplicate filing-event keys detected; these should be investigated as possible duplicate rows in the source or import pipeline.")
        status = "PASS_WITH_WARNINGS"
    if info_after_filed:
        warnings.append(f"{info_after_filed} information timestamps occur substantially after filed dates; inspect source mapping.")
        status = "PASS_WITH_WARNINGS"

    return {
        "rows": rows,
        "chunks": chunks,
        "unique_ciks": len(unique_ciks),
        "required_nulls": nulls,
        "duplicate_event_keys": duplicate_keys,
        "exact_duplicate_event_keys": exact_duplicate_keys,
        "economic_duplicate_event_keys": economic_duplicate_keys,
        "information_time_parse_failures": bad_time,
        "information_after_filed_gt_1d": info_after_filed,
        "information_time_start": None if min_info is None else min_info.isoformat(),
        "information_time_end": None if max_info is None else max_info.isoformat(),
        "metrics": metrics,
        "status": status,
        "warnings": warnings,
    }
