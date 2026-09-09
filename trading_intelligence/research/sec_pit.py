from __future__ import annotations

from pathlib import Path

import pandas as pd


SEC_FEATURE_COLUMNS = (
    "cik",
    "metric",
    "unit",
    "period_type",
    "start",
    "end",
    "val_num",
    "information_time",
    "filed",
    "form",
    "accn",
    "tag",
)


def load_canonical_sec_observations(
    path: str | Path | pd.DataFrame,
    *,
    ciks: set[int] | None = None,
    metrics: set[str] | None = None,
) -> pd.DataFrame:
    """Load canonical PIT SEC observations without collapsing economic periods."""
    if isinstance(path, pd.DataFrame):
        frame = path.copy()
    else:
        frame = pd.read_csv(path, low_memory=False)

    required = {
        "cik",
        "metric",
        "unit",
        "start",
        "end",
        "information_time",
        "val_num",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"SEC observations missing columns: {missing}")

    out = frame.copy()
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    out["information_time"] = pd.to_datetime(
        out["information_time"], utc=True, errors="coerce"
    )
    out["start"] = pd.to_datetime(out["start"], utc=True, errors="coerce")
    out["end"] = pd.to_datetime(out["end"], utc=True, errors="coerce")
    out["val_num"] = pd.to_numeric(out["val_num"], errors="coerce")

    out = out.dropna(
        subset=["cik", "metric", "unit", "information_time", "end", "val_num"]
    ).copy()

    if "period_type" not in out.columns:
        out["period_type"] = "duration"

    if ciks is not None:
        out = out[out["cik"].isin(sorted(ciks))].copy()

    if metrics is not None:
        out = out[out["metric"].isin(sorted(metrics))].copy()

    return out.sort_values(
        [
            "cik",
            "metric",
            "unit",
            "period_type",
            "start",
            "end",
            "information_time",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def align_sec_asof(
    decisions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Align SEC observations to decisions using strict point-in-time semantics.

    For each decision, every economically distinct SEC period is retained.
    Within each CIK/metric/unit/period identity, the latest observation whose
    information_time is <= decision_time is selected.
    """
    required = {"cik", "decision_time"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"Decision frame missing columns: {missing}")

    dec = decisions.copy()
    dec["cik"] = pd.to_numeric(dec["cik"], errors="coerce").astype("Int64")
    dec["decision_time"] = pd.to_datetime(
        dec["decision_time"], utc=True, errors="coerce"
    )
    dec = dec.dropna(subset=["cik", "decision_time"]).copy()

    obs = load_canonical_sec_observations(observations)

    if dec.empty:
        return pd.DataFrame()

    if obs.empty:
        result = dec.copy()
        result["metric"] = pd.NA
        result["unit"] = pd.NA
        result["period_type"] = pd.NA
        result["start"] = pd.NaT
        result["end"] = pd.NaT
        result["val_num"] = pd.NA
        result["information_time"] = pd.NaT
        result["available"] = False
        return result

    frames: list[pd.DataFrame] = []

    period_keys = [
        "cik",
        "metric",
        "unit",
        "period_type",
        "start",
        "end",
    ]

    for period_key, group in obs.groupby(
        period_keys,
        dropna=False,
        sort=False,
    ):
        cik = int(period_key[0])

        dgrp = dec[dec["cik"] == cik].copy()
        if dgrp.empty:
            continue

        group = group.sort_values(
            "information_time",
            kind="mergesort",
        ).copy()

        dgrp = dgrp.sort_values(
            "decision_time",
            kind="mergesort",
        ).copy()

        payload_columns = [
            c
            for c in [
                "information_time",
                "val_num",
                "filed",
                "form",
                "accn",
                "tag",
                "entity_name",
            ]
            if c in group.columns
        ]

        out = pd.merge_asof(
            dgrp,
            group[payload_columns],
            left_on="decision_time",
            right_on="information_time",
            direction="backward",
            allow_exact_matches=True,
        )

        for column_name, value in zip(period_keys[1:], period_key[1:]):
            out[column_name] = value

        out["available"] = (
            out["information_time"].notna()
            & (out["information_time"] <= out["decision_time"])
        )

        frames.append(out)

    if not frames:
        result = dec.copy()
        result["metric"] = pd.NA
        result["unit"] = pd.NA
        result["period_type"] = pd.NA
        result["start"] = pd.NaT
        result["end"] = pd.NaT
        result["val_num"] = pd.NA
        result["information_time"] = pd.NaT
        result["available"] = False
        return result

    result = pd.concat(frames, ignore_index=True, sort=False)

    leaked = result["information_time"].notna() & (
        result["information_time"] > result["decision_time"]
    )
    if leaked.any():
        raise AssertionError(
            f"SEC PIT leakage detected in {int(leaked.sum())} row(s)."
        )

    preferred = [
        "cik",
        "decision_time",
        "metric",
        "unit",
        "period_type",
        "start",
        "end",
        "val_num",
        "information_time",
        "filed",
        "form",
        "accn",
        "tag",
        "entity_name",
        "available",
    ]

    cols = [c for c in preferred if c in result.columns]
    cols += [c for c in result.columns if c not in cols]

    return result[cols].sort_values(
        [
            "cik",
            "decision_time",
            "metric",
            "unit",
            "period_type",
            "start",
            "end",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def build_sec_feature_panel(
    decisions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Return a period-preserving PIT SEC feature panel."""
    aligned = align_sec_asof(decisions, observations)

    if aligned.empty:
        return aligned

    aligned["feature_available"] = (
        aligned["available"].fillna(False).astype(bool)
    )

    aligned["observation_age_days"] = (
        aligned["decision_time"] - aligned["information_time"]
    ).dt.total_seconds() / 86400.0

    return aligned
