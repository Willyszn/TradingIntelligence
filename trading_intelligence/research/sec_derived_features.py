from __future__ import annotations

import numpy as np
import pandas as pd


def _normalise(observations: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cik",
        "metric",
        "unit",
        "start",
        "end",
        "information_time",
        "val_num",
        "duration_kind",
        "reporting_kind",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"SEC observations missing columns: {missing}")

    out = observations.copy()

    for col in ["start", "end", "information_time"]:
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    out["val_num"] = pd.to_numeric(out["val_num"], errors="coerce")
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")

    return out.dropna(
        subset=["cik", "metric", "unit", "end", "information_time", "val_num"]
    ).copy()


def _latest_revision_per_period(group: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest known revision for each decision-information timestamp."""
    period_keys = [
        "cik",
        "metric",
        "unit",
        "reporting_kind",
        "duration_kind",
        "start",
        "end",
    ]

    work = group.sort_values(
        period_keys + ["information_time"],
        kind="mergesort",
    ).copy()

    return (
        work.groupby(
            period_keys,
            dropna=False,
            as_index=False,
        )
        .tail(1)
        .sort_values(
            ["cik", "metric", "unit", "end", "information_time"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def add_qoq_growth(observations: pd.DataFrame) -> pd.DataFrame:
    """Add quarter-over-quarter growth using only latest-known prior quarters."""
    out = _normalise(observations)

    out["qoq_growth"] = np.nan

    for keys, group in out.groupby(
        ["cik", "metric", "unit"],
        dropna=False,
        sort=False,
    ):
        q = group[group["duration_kind"].eq("quarter")].copy()
        if q.empty:
            continue

        q = _latest_revision_per_period(q)

        # Use period-end ordering, then compare each quarter to the prior
        # economic quarter. Revisions remain anchored to information_time.
        q = q.sort_values(
            ["end", "information_time"],
            kind="mergesort",
        ).copy()

        values = q["val_num"].to_numpy(dtype=float)
        prior = q["val_num"].shift(1)

        growth = np.where(
            prior.notna() & prior.ne(0),
            q["val_num"] / prior - 1.0,
            np.nan,
        )

        out.loc[q.index, "qoq_growth"] = growth

    return out


def add_yoy_growth(observations: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year growth for quarterly observations."""
    out = _normalise(observations)

    out["yoy_growth"] = np.nan

    for _, group in out.groupby(
        ["cik", "metric", "unit"],
        dropna=False,
        sort=False,
    ):
        q = group[group["duration_kind"].eq("quarter")].copy()
        if q.empty:
            continue

        q = _latest_revision_per_period(q)
        q = q.sort_values(["end", "information_time"], kind="mergesort")

        # Four reported quarters back within the same economic series.
        prior = q["val_num"].shift(4)

        growth = np.where(
            prior.notna() & prior.ne(0),
            q["val_num"] / prior - 1.0,
            np.nan,
        )

        out.loc[q.index, "yoy_growth"] = growth

    return out


def add_ttm(observations: pd.DataFrame) -> pd.DataFrame:
    """Add TTM values from four consecutive quarterly observations.

    TTM is calculated independently for each observation revision.
    A quarter contributes only when that quarter was already known by
    the current observation's information_time.
    """
    out = _normalise(observations)

    out["ttm_value"] = np.nan

    for _, group in out.groupby(
        ["cik", "metric", "unit"],
        dropna=False,
        sort=False,
    ):
        q = group[group["duration_kind"].eq("quarter")].copy()
        if q.empty:
            continue

        q = q.sort_values(
            ["end", "information_time"],
            kind="mergesort",
        ).copy()

        rows = []

        # Each source row is treated as its own PIT decision timestamp.
        for idx, row in q.iterrows():
            known = q[q["information_time"] <= row["information_time"]].copy()

            latest_by_period = (
                known.sort_values(
                    ["end", "information_time"],
                    kind="mergesort",
                )
                .groupby(
                    ["start", "end"],
                    dropna=False,
                    as_index=False,
                )
                .tail(1)
                .sort_values("end", kind="mergesort")
            )

            latest_four = latest_by_period.tail(4)

            if len(latest_four) != 4:
                value = np.nan
            else:
                ends = latest_four["end"].tolist()

                # Require four genuinely sequential quarters.
                gaps = [
                    (ends[i] - ends[i - 1]).days
                    for i in range(1, len(ends))
                ]

                sequential = all(70 <= gap <= 110 for gap in gaps)

                value = (
                    float(latest_four["val_num"].sum())
                    if sequential
                    else np.nan
                )

            rows.append((idx, value))

        for idx, value in rows:
            out.loc[idx, "ttm_value"] = value

    return out


def add_all_derived_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Add QoQ, YoY, and TTM features while retaining raw SEC observations."""
    out = add_qoq_growth(observations)
    out = add_yoy_growth(out)
    out = add_ttm(out)

    out["ttm_available"] = out["ttm_value"].notna()

    return out
