
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
        raise ValueError(
            f"SEC observations missing columns: {missing}"
        )

    out = observations.copy()

    out["cik"] = pd.to_numeric(
        out["cik"],
        errors="coerce",
    ).astype("Int64")

    for column in [
        "start",
        "end",
        "information_time",
    ]:
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
    observations: pd.DataFrame,
) -> pd.DataFrame:
    work = observations.copy()

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


def _adjacent_quarter(
    current_end: pd.Timestamp,
    previous_end: pd.Timestamp,
) -> bool:
    if pd.isna(current_end) or pd.isna(previous_end):
        return False

    gap = (current_end - previous_end).days

    return 70 <= gap <= 110


def _year_apart(
    current_end: pd.Timestamp,
    previous_end: pd.Timestamp,
) -> bool:
    if pd.isna(current_end) or pd.isna(previous_end):
        return False

    gap = (current_end - previous_end).days

    return 300 <= gap <= 430


def _period_mask(
    out: pd.DataFrame,
    row: pd.Series,
) -> pd.Series:
    mask = (
        (out["cik"] == row["cik"])
        & (out["metric"] == row["metric"])
        & (out["unit"] == row["unit"])
        & (
            out["reporting_kind"]
            == row["reporting_kind"]
        )
        & (
            out["duration_kind"]
            == row["duration_kind"]
        )
        & (
            out["start"] == row["start"]
        )
        & (
            out["end"] == row["end"]
        )
        & (
            out["information_time"]
            == row["information_time"]
        )
    )

    return mask


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

        for position in range(1, len(q)):
            current = q.iloc[position]
            previous = q.iloc[position - 1]

            if not _adjacent_quarter(
                current["end"],
                previous["end"],
            ):
                continue

            denominator = previous["val_num"]

            if pd.isna(denominator) or denominator == 0:
                continue

            growth = (
                current["val_num"]
                / denominator
                - 1.0
            )

            out.loc[
                _period_mask(out, current),
                "qoq_growth",
            ] = float(growth)

    return out.reset_index(drop=True)


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

        if len(q) < 5:
            continue

        q = _latest_revision_per_period(q)

        for position in range(4, len(q)):
            current = q.iloc[position]
            previous = q.iloc[position - 4]

            if not _year_apart(
                current["end"],
                previous["end"],
            ):
                continue

            denominator = previous["val_num"]

            if pd.isna(denominator) or denominator == 0:
                continue

            growth = (
                current["val_num"]
                / denominator
                - 1.0
            )

            out.loc[
                _period_mask(out, current),
                "yoy_growth",
            ] = float(growth)

    return out.reset_index(drop=True)


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

        if len(q) < 4:
            continue

        q = _latest_revision_per_period(q)

        for position in range(3, len(q)):
            window = q.iloc[
                position - 3 : position + 1
            ]

            ends = window["end"].tolist()

            if not all(
                _adjacent_quarter(
                    ends[i],
                    ends[i - 1],
                )
                for i in range(1, len(ends))
            ):
                continue

            current = q.iloc[position]

            value = float(
                window["val_num"].sum()
            )

            out.loc[
                _period_mask(out, current),
                "ttm_value",
            ] = value

    out["ttm_available"] = (
        out["ttm_value"].notna()
    )

    return out.reset_index(drop=True)


def add_all_derived_features(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = add_qoq_growth(observations)
    out = add_yoy_growth(out)
    out = add_ttm(out)

    return out.reset_index(drop=True)
