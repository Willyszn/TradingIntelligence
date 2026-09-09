from __future__ import annotations

import pandas as pd


def load_fred_pit(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load FRED/ALFRED observations with conservative availability dates."""
    if isinstance(path, pd.DataFrame):
        frame = path.copy()
    else:
        frame = pd.read_csv(path, low_memory=False)

    required = {
        "series_id",
        "date",
        "value",
        "realtime_start",
        "realtime_end",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FRED data missing columns: {missing}")

    out = frame.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["realtime_start"] = pd.to_datetime(
        out["realtime_start"],
        errors="coerce",
    )
    out["realtime_end"] = pd.to_datetime(
        out["realtime_end"],
        errors="coerce",
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if "available_date_conservative" not in out.columns:
        out["available_date_conservative"] = (
            out["realtime_start"] + pd.Timedelta(days=1)
        )
    else:
        out["available_date_conservative"] = pd.to_datetime(
            out["available_date_conservative"],
            errors="coerce",
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

    return out.sort_values(
        ["series_id", "date", "realtime_start"],
        kind="mergesort",
    ).reset_index(drop=True)


def latest_vintage_asof(
    observations: pd.DataFrame,
    *,
    decision_time: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return the latest vintage legally available by decision_time."""
    out = load_fred_pit(observations)

    decision = pd.Timestamp(decision_time)

    if decision.tzinfo is not None:
        decision = decision.tz_convert(None)
    else:
        decision = decision.tz_localize(None)

    out = out[
        out["available_date_conservative"] <= decision
    ].copy()

    if out.empty:
        return out

    out = (
        out.sort_values(
            ["series_id", "date", "realtime_start"],
            kind="mergesort",
        )
        .groupby(
            ["series_id", "date"],
            as_index=False,
            dropna=False,
        )
        .tail(1)
    )

    return out.sort_values(
        ["series_id", "date"],
        kind="mergesort",
    ).reset_index(drop=True)


def align_fred_asof(
    decisions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Align each FRED series to the latest economic observation available
    at each decision time, then select the latest vintage for that
    observation that was itself conservatively available.

    This is deliberately a two-stage alignment:

    1. select the latest economic observation date <= decision date;
    2. for that exact observation date, select the latest vintage whose
       conservative availability <= decision date.

    This prevents a future revision from hiding an older usable vintage.
    """
    required = {"decision_time"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"Decision frame missing columns: {missing}")

    dec = decisions.copy()

    dec["decision_time"] = pd.to_datetime(
        dec["decision_time"],
        utc=True,
        errors="coerce",
    )

    dec = dec.dropna(subset=["decision_time"]).copy()

    obs = load_fred_pit(observations)

    dec["_fred_decision_date"] = (
        dec["decision_time"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    frames: list[pd.DataFrame] = []

    for series_id, group in obs.groupby(
        "series_id",
        sort=False,
        dropna=False,
    ):
        economic_dates = (
            group[["date"]]
            .drop_duplicates()
            .sort_values("date", kind="mergesort")
            .reset_index(drop=True)
        )

        selected = pd.merge_asof(
            dec.sort_values("_fred_decision_date", kind="mergesort"),
            economic_dates,
            left_on="_fred_decision_date",
            right_on="date",
            direction="backward",
            allow_exact_matches=True,
        )

        selected["_selected_date"] = selected["date"]
        selected = selected.drop(
            columns=["date"],
            errors="ignore",
        )

        vintage = group.copy()
        vintage["_selected_date"] = vintage["date"]

        vintage = vintage.sort_values(
            [
                "_selected_date",
                "available_date_conservative",
                "realtime_start",
            ],
            kind="mergesort",
        )

        out = pd.merge_asof(
            selected.sort_values(
                "_fred_decision_date",
                kind="mergesort",
            ),
            vintage[
                [
                    "_selected_date",
                    "value",
                    "realtime_start",
                    "realtime_end",
                    "available_date_conservative",
                ]
            ].sort_values(
                "_selected_date",
                kind="mergesort",
            ),
            left_on="_fred_decision_date",
            right_on="_selected_date",
            direction="backward",
            allow_exact_matches=True,
        )

        # The merge above is intentionally not used for vintage selection:
        # availability must be <= decision date for the selected economic
        # observation. Perform that exact selection per selected date.
        out["series_id"] = series_id

        selected_rows = []

        for _, row in selected.iterrows():
            economic_date = row["_selected_date"]

            if pd.isna(economic_date):
                selected_rows.append(
                    {
                        "decision_time": row["decision_time"],
                        "series_id": series_id,
                        "value": pd.NA,
                        "realtime_start": pd.NaT,
                        "realtime_end": pd.NaT,
                        "available_date_conservative": pd.NaT,
                        "available": False,
                    }
                )
                continue

            candidates = group[
                (group["date"] == economic_date)
                & (
                    group["available_date_conservative"]
                    <= row["_fred_decision_date"]
                )
            ].copy()

            if candidates.empty:
                selected_rows.append(
                    {
                        "decision_time": row["decision_time"],
                        "series_id": series_id,
                        "value": pd.NA,
                        "realtime_start": pd.NaT,
                        "realtime_end": pd.NaT,
                        "available_date_conservative": pd.NaT,
                        "available": False,
                    }
                )
                continue

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

            selected_rows.append(
                {
                    "decision_time": row["decision_time"],
                    "series_id": series_id,
                    "value": candidate["value"],
                    "realtime_start": candidate["realtime_start"],
                    "realtime_end": candidate["realtime_end"],
                    "available_date_conservative": (
                        candidate["available_date_conservative"]
                    ),
                    "available": True,
                }
            )

        aligned = pd.DataFrame(selected_rows)

        # Preserve any decision-side columns.
        decision_cols = [
            c for c in dec.columns
            if c not in {"_fred_decision_date"}
        ]

        if decision_cols:
            aligned = dec[decision_cols].merge(
                aligned[
                    [
                        "decision_time",
                        "series_id",
                        "value",
                        "realtime_start",
                        "realtime_end",
                        "available_date_conservative",
                        "available",
                    ]
                ],
                on="decision_time",
                how="left",
                sort=False,
            )

        frames.append(aligned)

    if not frames:
        result = dec.copy()
        result["series_id"] = pd.NA
        result["value"] = pd.NA
        result["available_date_conservative"] = pd.NaT
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

    result["available"] = result["available"].fillna(False).astype(bool)

    leaked = result["available"] & (
        pd.to_datetime(
            result["available_date_conservative"],
            errors="coerce",
        )
        > result["decision_time"].dt.tz_convert(None).dt.normalize()
    )

    if leaked.any():
        raise AssertionError(
            f"FRED PIT leakage detected in {int(leaked.sum())} row(s)."
        )

    return result.drop(
        columns=["_fred_decision_date"],
        errors="ignore",
    ).sort_values(
        ["decision_time", "series_id"],
        kind="mergesort",
    ).reset_index(drop=True)
