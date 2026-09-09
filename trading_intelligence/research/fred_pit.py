
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_fred_pit(
    path: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Load FRED/ALFRED observations with conservative availability."""
    if isinstance(path, pd.DataFrame):
        frame = path.copy()
    else:
        frame = pd.read_csv(
            path,
            low_memory=False,
        )

    required = {
        "series_id",
        "date",
        "value",
        "realtime_start",
        "realtime_end",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"FRED data missing columns: {missing}"
        )

    out = frame.copy()

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    )

    out["realtime_start"] = pd.to_datetime(
        out["realtime_start"],
        errors="coerce",
    )

    out["realtime_end"] = pd.to_datetime(
        out["realtime_end"],
        errors="coerce",
    )

    out["value"] = pd.to_numeric(
        out["value"],
        errors="coerce",
    )

    if "available_date_conservative" not in out.columns:
        out["available_date_conservative"] = (
            out["realtime_start"]
            + pd.Timedelta(days=1)
        )
    else:
        out["available_date_conservative"] = pd.to_datetime(
            out["available_date_conservative"],
            errors="coerce",
        )

    return (
        out.dropna(
            subset=[
                "series_id",
                "date",
                "value",
                "realtime_start",
                "available_date_conservative",
            ]
        )
        .sort_values(
            [
                "series_id",
                "date",
                "realtime_start",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def latest_vintage_asof(
    observations: pd.DataFrame,
    *,
    decision_time: str | pd.Timestamp,
) -> pd.DataFrame:
    out = load_fred_pit(observations)

    decision = pd.Timestamp(decision_time)

    if decision.tzinfo is not None:
        decision = decision.tz_convert(None)
    else:
        decision = decision.tz_localize(None)

    out = out[
        out["available_date_conservative"]
        <= decision
    ].copy()

    if out.empty:
        return out

    return (
        out.sort_values(
            [
                "series_id",
                "date",
                "realtime_start",
            ],
            kind="mergesort",
        )
        .groupby(
            ["series_id", "date"],
            as_index=False,
            dropna=False,
        )
        .tail(1)
        .sort_values(
            ["series_id", "date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def align_fred_asof(
    decisions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Align each series to the latest legally available observation.

    Important ordering:

    1. filter vintages by conservative availability;
    2. choose the latest economic observation date;
    3. choose the latest vintage for that date.

    All decision-side columns are preserved.
    """
    if "decision_time" not in decisions.columns:
        raise ValueError(
            "Decision frame missing columns: ['decision_time']"
        )

    dec = decisions.copy()

    dec["decision_time"] = pd.to_datetime(
        dec["decision_time"],
        utc=True,
        errors="coerce",
    )

    dec = dec.dropna(
        subset=["decision_time"]
    ).copy()

    dec["_fred_decision_date"] = (
        dec["decision_time"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    obs = load_fred_pit(observations)

    frames = []

    for series_id, group in obs.groupby(
        "series_id",
        sort=False,
        dropna=False,
    ):
        rows = []

        for _, decision in dec.iterrows():
            decision_date = decision["_fred_decision_date"]

            available = group[
                group["available_date_conservative"]
                <= decision_date
            ].copy()

            row = decision.drop(
                labels=["_fred_decision_date"],
                errors="ignore",
            ).to_dict()

            row["series_id"] = series_id

            if available.empty:
                row.update(
                    {
                        "value": pd.NA,
                        "realtime_start": pd.NaT,
                        "realtime_end": pd.NaT,
                        "available_date_conservative": pd.NaT,
                        "observation_date": pd.NaT,
                        "available": False,
                    }
                )

                rows.append(row)
                continue

            latest_date = available["date"].max()

            candidates = available[
                available["date"] == latest_date
            ].copy()

            candidate = (
                candidates.sort_values(
                    [
                        "available_date_conservative",
                        "realtime_start",
                    ],
                    kind="mergesort",
                )
                .iloc[-1]
            )

            row.update(
                {
                    "value": candidate["value"],
                    "realtime_start": candidate[
                        "realtime_start"
                    ],
                    "realtime_end": candidate[
                        "realtime_end"
                    ],
                    "available_date_conservative": candidate[
                        "available_date_conservative"
                    ],
                    "observation_date": candidate["date"],
                    "available": True,
                }
            )

            rows.append(row)

        frames.append(pd.DataFrame(rows))

    if not frames:
        result = dec.copy()
        result["series_id"] = pd.NA
        result["value"] = pd.NA
        result["available"] = False

        return result.drop(
            columns=["_fred_decision_date"],
            errors="ignore",
        )

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    result["available"] = (
        result["available"]
        .fillna(False)
        .astype(bool)
    )

    decision_dates = (
        result["decision_time"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    available_dates = pd.to_datetime(
        result["available_date_conservative"],
        errors="coerce",
    )

    leaked = (
        result["available"]
        & available_dates.notna()
        & (available_dates > decision_dates)
    )

    if leaked.any():
        raise AssertionError(
            "FRED PIT leakage detected in "
            f"{int(leaked.sum())} row(s)."
        )

    return result.sort_values(
        [
            "decision_time",
            "series_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
