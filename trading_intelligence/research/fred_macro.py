from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

from trading_intelligence.env import load_project_env


FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_MAX_DATE = "2262-04-11"

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
        raise RuntimeError(
            "FRED_API_KEY is not set. "
            "Set it in the environment; do not place it in project files."
        )

    return key.strip()


def _normalise_dates(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    out = frame.copy()

    for column in columns:
        out[column] = (
            out[column]
            .astype(str)
            .replace({"9999-12-31": FRED_MAX_DATE})
        )

        out[column] = pd.to_datetime(
            out[column],
            utc=True,
            errors="coerce",
            format="mixed",
        )

    return out


def _parse_vintage_observations(
    series_id: str,
    observations: list[dict],
) -> list[dict]:
    """Convert FRED output_type=2 wide rows into long PIT observations."""

    pattern = re.compile(
        rf"^{re.escape(series_id)}_(\d{{8}})$"
    )

    rows: list[dict] = []

    for item in observations:
        observation_date = item.get("date")

        if not observation_date:
            continue

        for key, raw_value in item.items():
            match = pattern.match(str(key))

            if not match:
                continue

            if raw_value in (None, "", "."):
                continue

            vintage_date = pd.to_datetime(
                match.group(1),
                format="%Y%m%d",
                utc=True,
                errors="coerce",
            )

            value = pd.to_numeric(
                raw_value,
                errors="coerce",
            )

            if pd.isna(vintage_date) or pd.isna(value):
                continue

            rows.append(
                {
                    "series_id": series_id,
                    "observation_date": observation_date,
                    "value": float(value),
                    "realtime_start": vintage_date,
                    "realtime_end": vintage_date,
                }
            )

    return rows


def fetch_series_vintages(
    series_id: str,
    api_key: str,
    observation_start: str = "1900-01-01",
    observation_end: str = "9999-12-31",
    vintage_start: str = "2023-01-01",
    vintage_end: str | None = None,
    request_timeout: int = 60,
    sleep_seconds: float = 0.25,
    session: requests.Session | None = None,
) -> pd.DataFrame:

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
                "output_type": 2,
                "realtime_start": vintage_start,
                "observation_start": observation_start,
                "sort_order": "asc",
                "limit": 100000,
                "offset": offset,
            }

            if vintage_end:
                params["realtime_end"] = vintage_end

            if (
                observation_end
                and observation_end != "9999-12-31"
            ):
                params["observation_end"] = observation_end

            response = sess.get(
                FRED_API_URL,
                params=params,
                timeout=request_timeout,
            )
            response.raise_for_status()

            payload = response.json()
            batch = payload.get("observations", [])

            if not batch:
                break

            rows.extend(
                _parse_vintage_observations(
                    series_id,
                    batch,
                )
            )

            if len(batch) < 2000:
                break

            offset += len(batch)
            time.sleep(sleep_seconds)

    finally:
        if own_session:
            sess.close()

    columns = [
        "series_id",
        "observation_date",
        "value",
        "realtime_start",
        "realtime_end",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)

    out = _normalise_dates(
        out,
        [
            "observation_date",
            "realtime_start",
            "realtime_end",
        ],
    )

    out["value"] = pd.to_numeric(
        out["value"],
        errors="coerce",
    )

    out = out.dropna(
        subset=[
            "observation_date",
            "value",
            "realtime_start",
            "realtime_end",
        ]
    ).copy()

    return (
        out.sort_values(
            [
                "observation_date",
                "realtime_start",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            [
                "series_id",
                "observation_date",
                "value",
                "realtime_start",
                "realtime_end",
            ]
        )
        .reset_index(drop=True)
    )


def download_macro_vintages(
    output_csv: str | Path,
    series: dict[str, str] | None = None,
    api_key: str | None = None,
    observation_start: str = "1900-01-01",
    observation_end: str = "9999-12-31",
    vintage_start: str = "2023-01-01",
    vintage_end: str | None = None,
) -> dict:

    key = _api_key(api_key)
    series = series or DEFAULT_SERIES

    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    vintages: dict[str, int] = {}

    for series_id in series:
        frame = fetch_series_vintages(
            series_id,
            key,
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=vintage_start,
            vintage_end=vintage_end,
        )

        if not frame.empty:
            frame["feature_name"] = series[series_id]
            frames.append(frame)

            vintages[series_id] = int(
                frame["realtime_start"].nunique()
            )
        else:
            vintages[series_id] = 0

        counts[series_id] = int(len(frame))

    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )

    if not out.empty:
        out = out.sort_values(
            [
                "series_id",
                "observation_date",
                "realtime_start",
            ],
            kind="mergesort",
        )

        out["available_date_conservative"] = (
            out["realtime_start"]
            + pd.Timedelta(days=1)
        ).dt.normalize()

        out = out.reset_index(drop=True)

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        output,
        index=False,
    )

    return {
        "series_count": len(series),
        "series_rows": counts,
        "series_vintages": vintages,
        "total_rows": int(len(out)),
        "vintage_start": vintage_start,
        "vintage_end": vintage_end,
        "output": str(output),
    }


def audit_macro_vintages(
    df: pd.DataFrame,
) -> dict:

    required = [
        "series_id",
        "observation_date",
        "value",
        "realtime_start",
        "realtime_end",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    out = _normalise_dates(
        df,
        [
            "observation_date",
            "realtime_start",
            "realtime_end",
        ],
    )

    out["value"] = pd.to_numeric(
        out["value"],
        errors="coerce",
    )

    required_nulls = {
        column: int(out[column].isna().sum())
        for column in required
    }

    realtime_inversion_rows = int(
        (
            out["realtime_end"]
            < out["realtime_start"]
        ).sum()
    )

    duplicate_rows = int(
        out.duplicated(
            required,
            keep=False,
        ).sum()
    )

    vintage_count = int(
        out["realtime_start"].nunique()
    )

    revision_pairs = (
        out.groupby(
            [
                "series_id",
                "observation_date",
            ],
            dropna=False,
        )["realtime_start"]
        .nunique()
    )

    revised_observation_keys = int(
        (revision_pairs > 1).sum()
    )

    result = {
        "required_nulls": required_nulls,
        "realtime_inversion_rows": realtime_inversion_rows,
        "duplicate_rows": duplicate_rows,
        "rows": int(len(out)),
        "series": int(out["series_id"].nunique()),
        "unique_vintages": vintage_count,
        "revised_observation_keys": revised_observation_keys,
    }

    result["status"] = (
        "PASS"
        if (
            sum(required_nulls.values()) == 0
            and realtime_inversion_rows == 0
            and duplicate_rows == 0
            and vintage_count > 1
        )
        else "FAIL"
    )

    return result
