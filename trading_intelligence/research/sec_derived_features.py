
from __future__ import annotations

import numpy as np
import pandas as pd


PERIOD_KEYS = [
    "cik",
    "metric",
    "unit",
    "reporting_kind",
    "duration_kind",
    "start",
    "end",
]


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

    out["cik"] = pd.to_numeric(
        out["cik"],
        errors="coerce",
    ).astype("Int64")

    for column in ["start", "end", "information_time"]:
        out[column] = pd.to_datetime(
            out[column],
            utc=True,
            errors="coerce",
        )

    out["val_num"] = pd.to_numeric(
        out["val_num"],
        errors="coerce",
    )

    return (
        out.dropna(
            subset=[
                "cik",
                "metric",
                "unit",
                "end",
                "information_time",
                "val_num",
            ]
        )
        .copy()
        .reset_index(drop=True)
    )


def _latest_revision_per_period(
    group: pd.DataFrame,
) -> pd.DataFrame:
    work = group.copy().reset_index(drop=True)

    work = work.sort_values(
        PERIOD_KEYS + ["information_time"],
        kind="mergesort",
    )

    return (
        work.groupby(
            PERIOD_KEYS,
            dropna=False,
            as_index=False,
        )
        .tail(1)
        .sort_values(
            ["end", "information_time"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def add_qoq_growth(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = _normalise(observations)
    out["qoq_growth"] = np.nan

    for _, group in out.groupby(
        ["cik", "metric", "unit"],
        dropna=False,
        sort=False,
    ):
        q = group[
            group["duration_kind"].eq("quarter")
        ].copy()

        if q.empty:
            continue

        q = _latest_revision_per_period(q)
        q = q.sort_values(
            "end",
            kind="mergesort",
        ).reset_index(drop=True)

        prior = q["val_num"].shift(1)

        growth = np.where(
            prior.notna() & prior.ne(0),
            q["val_num"] / prior - 1.0,
            np.nan,
        )

        # Match by economic period identity, not DataFrame index.
        for source_idx, value in zip(
            q.index,
            growth,
            strict=True,
        ):
            start = q.loc[source_idx, "start"]
            end = q.loc[source_idx, "end"]

            mask = (
                (out["cik"] == q.loc[source_idx, "cik"])
                & (out["metric"] == q.loc[source_idx, "metric"])
                & (out["unit"] == q.loc[source_idx, "unit"])
                & (out["start"] == start)
                & (out["end"] == end)
                & (
                    out["duration_kind"]
                    == q.loc[source_idx, "duration_kind"]
                )
                & (
                    out["reporting_kind"]
                    == q.loc[source_idx, "reporting_kind"]
                )
            )

            out.loc[mask, "qoq_growth"] = value

    return out


def add_yoy_growth(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = _normalise(observations)
    out["yoy_growth"] = np.nan

    for _, group in out.groupby(
        ["cik", "metric", "unit"],
        dropna=False,
        sort=False,
    ):
        q = group[
            group["duration_kind"].eq("quarter")
        ].copy()

        if q.empty:
            continue

        q = _latest_revision_per_period(q)
        q = q.sort_values(
            "end",
            kind="mergesort",
        ).reset_index(drop=True)

        prior = q["val_num"].shift(4)

        growth = np.where(
            prior.notna() & prior.ne(0),
            q["val_num"] / prior - 1.0,
            np.nan,
        )

        for source_idx, value in zip(
            q.index,
            growth,
            strict=True,
        ):
            start = q.loc[source_idx, "start"]
            end = q.loc[source_idx, "end"]

            mask = (
                (out["cik"] == q.loc[source_idx, "cik"])
                & (out["metric"] == q.loc[source_idx, "metric"])
                & (out["unit"] == q.loc[source_idx, "unit"])
                & (out["start"] == start)
                & (out["end"] == end)
                & (
                    out["duration_kind"]
                    == q.loc[source_idx, "duration_kind"]
                )
                & (
                    out["reporting_kind"]
                    == q.loc[source_idx, "reporting_kind"]
                )
            )

            out.loc[mask, "yoy_growth"] = value

    return out


def add_ttm(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = _normalise(observations)
    out["ttm_value"] = np.nan

    for _, group in out.groupby(
        ["cik", "metric", "unit"],
        dropna=False,
        sort=False,
    ):
        q = group[
            group["duration_kind"].eq("quarter")
        ].copy()

        if q.empty:
            continue

        q = q.sort_values(
            ["end", "information_time"],
            kind="mergesort",
        ).reset_index(drop=True)

        for _, row in q.iterrows():
            known = q[
                q["information_time"] <= row["information_time"]
            ].copy()

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
                .sort_values(
                    "end",
                    kind="mergesort",
                )
                .reset_index(drop=True)
            )

            latest_four = latest_by_period.tail(4)

            value = np.nan

            if len(latest_four) == 4:
                ends = latest_four["end"].tolist()

                gaps = [
                    (ends[i] - ends[i - 1]).days
                    for i in range(1, len(ends))
                ]

                sequential = all(
                    70 <= gap <= 110
                    for gap in gaps
                )

                if sequential:
                    value = float(
                        latest_four["val_num"].sum()
                    )

            mask = (
                (out["cik"] == row["cik"])
                & (out["metric"] == row["metric"])
                & (out["unit"] == row["unit"])
                & (out["start"] == row["start"])
                & (out["end"] == row["end"])
                & (
                    out["duration_kind"]
                    == row["duration_kind"]
                )
                & (
                    out["reporting_kind"]
                    == row["reporting_kind"]
                )
                & (
                    out["information_time"]
                    == row["information_time"]
                )
            )

            out.loc[mask, "ttm_value"] = value

    return out


def add_all_derived_features(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = add_qoq_growth(observations)
    out = add_yoy_growth(out)
    out = add_ttm(out)

    out["ttm_available"] = out["ttm_value"].notna()

    return out.reset_index(drop=True)
