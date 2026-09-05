from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

from trading_intelligence.env import load_project_env

import pandas as pd
import requests

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_MAX_DATE = "2262-04-11"  # pandas UTC datetime upper-bound sentinel for FRED open-ended realtime_end=9999-12-31

DEFAULT_SERIES = {
    "EFFR": "effective_federal_funds_rate",
    "DGS2": "treasury_2y",
    "DGS10": "treasury_10y",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate",
    "GDPC1": "real_gdp",
    "BAMLH0A0HYM2": "high_yield_spread",
    "VIXCLS": "vix",
    "DTWEXBGS": "broad_dollar_index",
}


def _api_key(api_key: str | None = None) -> str:
    load_project_env()
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set. Set it in the environment; do not place it in project files.")
    return key.strip()


def fetch_series_vintages(
    series_id: str,
    api_key: str,
    observation_start: str = "1900-01-01",
    observation_end: str = "9999-12-31",
    request_timeout: int = 60,
    sleep_seconds: float = 0.25,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch the full FRED real-time history for one series.

    FRED real-time periods preserve revision windows through realtime_start/end.
    We retain those fields rather than flattening to today's revised history.
    """
    own_session = session is None
    sess = session or requests.Session()
    rows: list[dict] = []
    offset = 0
    try:
        while True:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                # FRED documents 1776-07-04 as the earliest real-time
                # boundary. Omit open-ended realtime_end / observation_end
                # rather than sending 9999-12-31, which some API deployments
                # reject even though the documentation describes it as the
                # open-ended default.
                "observation_start": observation_start,
                "sort_order": "asc",
                "limit": 100000,
                "offset": offset,
            }
            if observation_end and observation_end != "9999-12-31":
                params["observation_end"] = observation_end
            r = sess.get(FRED_API_URL, params=params, timeout=request_timeout)
            r.raise_for_status()
            payload = r.json()
            batch = payload.get("observations", [])
            if not batch:
                break
            for item in batch:
                value = item.get("value")
                if value in (None, "", "."):
                    continue
                rows.append({
                    "series_id": series_id,
                    "observation_date": item.get("date"),
                    "value": value,
                    "realtime_start": item.get("realtime_start"),
                    "realtime_end": item.get("realtime_end"),
                })
            if len(batch) < 100000:
                break
            offset += len(batch)
            time.sleep(sleep_seconds)
    finally:
        if own_session:
            sess.close()

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["series_id", "observation_date", "value", "realtime_start", "realtime_end"])
    for c in ["observation_date", "realtime_start", "realtime_end"]:
        out[c] = out[c].astype(str).replace({"9999-12-31": FRED_MAX_DATE})
        out[c] = pd.to_datetime(out[c], utc=True, errors="coerce", format="mixed")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["observation_date", "value", "realtime_start", "realtime_end"]).copy()
    return out.sort_values(["observation_date", "realtime_start"], kind="mergesort").reset_index(drop=True)


def download_macro_vintages(
    output_csv: str | Path,
    series: dict[str, str] | None = None,
    api_key: str | None = None,
    observation_start: str = "1900-01-01",
    observation_end: str = "9999-12-31",
) -> dict:
    key = _api_key(api_key)
    series = series or DEFAULT_SERIES
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for series_id in series:
        df = fetch_series_vintages(
            series_id,
            key,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        if not df.empty:
            df["feature_name"] = series[series_id]
            frames.append(df)
        counts[series_id] = int(len(df))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["series_id", "observation_date", "realtime_start"], kind="mergesort")
        out["available_date_conservative"] = (out["realtime_start"] + pd.Timedelta(days=1)).dt.normalize()
        out = out.reset_index(drop=True)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return {
        "series_count": len(series),
        "series_rows": counts,
        "total_rows": int(len(out)),
        "output": str(output),
    }


def audit_macro_vintages(df: pd.DataFrame) -> dict:
    required = ["series_id", "observation_date", "value", "realtime_start", "realtime_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    for c in ["observation_date", "realtime_start", "realtime_end"]:
        out[c] = out[c].astype(str).replace({"9999-12-31": FRED_MAX_DATE})
        out[c] = pd.to_datetime(out[c], utc=True, errors="coerce", format="mixed")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    required_nulls = {c: int(out[c].isna().sum()) for c in required}
    realtime_inversion_rows = int((out["realtime_end"] < out["realtime_start"]).sum())
    duplicate_rows = int(out.duplicated(required, keep=False).sum())
    bad = {
        "required_nulls": required_nulls,
        "realtime_inversion_rows": realtime_inversion_rows,
        "duplicate_rows": duplicate_rows,
        "rows": int(len(out)),
        "series": int(out["series_id"].nunique()),
    }
    bad["status"] = "PASS" if sum(required_nulls.values()) == 0 and realtime_inversion_rows == 0 and duplicate_rows == 0 else "FAIL"
    return bad
