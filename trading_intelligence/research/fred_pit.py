
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "series_id",
    "date",
    "value",
    "realtime_start",
    "realtime_end",
}


def load_fred_pit(
    path: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        frame = path.copy()
    else:
        frame = pd.read_csv(
            path,
            low_memory=False,
        )

    missing = sorted(
        REQUIRED_COLUMNS - set(frame.columns)
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

    if "available_date_conservative" not in out:
        out["available_date_conservative"] = (
            out["realtime_start"]
            + pd.Timedelta(days=1)
        )
    else:
        out["available_date_conservative"] = (
            pd.to_datetime(
                out["available_date_conservative"],
                errors="coerce",
            )
        )

    out = out.dropna(
        subset=[
            "series_id",
            "date",
            "value",
            "realtime_start",
            "available_date_conservative",
        ]
    ).copy()

    return (
        out.sort_values(
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

    decision = pd.Timestamp(
        decision_time
    )

    if decision.tzinfo is not None:
        decision = decision.tz_convert(None)
    else:
        decision = decision.tz_localize(None)

    available = out[
        out["available_date_conservative"]
        <= decision
    ].copy()

    if available.empty:
        return available

    return (
        available.sort_values(
            [
                "series_id",
                "date",
                "realtime_start",
            ],
            kind="mergesort",
        )
        .groupby(
            ["series_id", "date"],
            dropna=False,
            as_index=False,
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
    if "decision_time" not in decisions:
        raise ValueError(
            "Decision frame missing columns: "
            "['decision_time']"
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

    if "_decision_id" not in dec.columns:
        dec["_decision_id"] = (
            range(len(dec))
        )

    dec["_decision_date"] = (
        dec["decision_time"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    obs = load_fred_pit(
        observations
    )

    frames = []

    for series_id, group in obs.groupby(
        "series_id",
        sort=False,
        dropna=False,
    ):
        group = group.sort_values(
            [
                "date",
                "available_date_conservative",
                "realtime_start",
            ],
            kind="mergesort",
        )

        # Cache one answer per unique decision date.
        decision_dates = (
            dec[
                [
                    "_decision_date"
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "_decision_date",
                kind="mergesort",
            )
        )

        state_rows = []

        for _, drow in decision_dates.iterrows():
            decision_date = drow[
                "_decision_date"
            ]

            available = group[
                group[
                    "available_date_conservative"
                ] <= decision_date
            ]

            if available.empty:
                state_rows.append(
                    {
                        "_decision_date": decision_date,
                        "value": pd.NA,
                        "realtime_start": pd.NaT,
                        "realtime_end": pd.NaT,
                        "available_date_conservative": pd.NaT,
                        "observation_date": pd.NaT,
                        "available": False,
                    }
                )
                continue

            latest_observation_date = (
                available["date"].max()
            )

            candidates = available[
                available["date"]
                == latest_observation_date
            ]

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

            state_rows.append(
                {
                    "_decision_date": decision_date,
                    "value": candidate["value"],
                    "realtime_start": candidate[
                        "realtime_start"
                    ],
                    "realtime_end": candidate[
                        "realtime_end"
                    ],
                    "available_date_conservative": (
                        candidate[
                            "available_date_conservative"
                        ]
                    ),
                    "observation_date": (
                        candidate["date"]
                    ),
                    "available": True,
                }
            )

        state = pd.DataFrame(
            state_rows
        )

        aligned = dec.merge(
            state,
            on="_decision_date",
            how="left",
            validate="many_to_one",
        )

        aligned["series_id"] = series_id

        frames.append(aligned)

    if not frames:
        result = dec.copy()
        result["series_id"] = pd.NA
        result["value"] = pd.NA
        result["available"] = False

        return result.drop(
            columns=["_decision_date"],
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
        & (
            available_dates
            > decision_dates
        )
    )

    if leaked.any():
        raise AssertionError(
            "FRED PIT leakage detected in "
            f"{int(leaked.sum())} row(s)."
        )

    return (
        result.drop(
            columns=["_decision_date"],
            errors="ignore",
        )
        .sort_values(
            [
                "_decision_id",
                "series_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
