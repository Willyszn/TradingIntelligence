from __future__ import annotations

import pandas as pd


INSTANT_METRICS = {
    "assets",
    "liabilities",
    "equity",
    "equity_including_nci",
    "cash",
    "debt_current",
    "debt_noncurrent",
    "debt_total",
}

DURATION_METRICS = {
    "revenue",
    "net_income",
    "eps_diluted",
}


def add_period_semantics(observations: pd.DataFrame) -> pd.DataFrame:
    """Add explicit SEC reporting-period semantics without collapsing observations."""
    required = {
        "metric",
        "start",
        "end",
        "information_time",
        "val_num",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"SEC observations missing columns: {missing}")

    out = observations.copy()

    out["start"] = pd.to_datetime(out["start"], utc=True, errors="coerce")
    out["end"] = pd.to_datetime(out["end"], utc=True, errors="coerce")
    out["information_time"] = pd.to_datetime(
        out["information_time"], utc=True, errors="coerce"
    )
    out["val_num"] = pd.to_numeric(out["val_num"], errors="coerce")

    out["period_days"] = (
        out["end"] - out["start"]
    ).dt.total_seconds() / 86400.0

    instant_mask = (
        out["start"].isna()
        | out["metric"].isin(INSTANT_METRICS)
    )

    out["reporting_kind"] = "duration"
    out.loc[instant_mask, "reporting_kind"] = "instant"

    duration_mask = out["reporting_kind"].eq("duration")

    out["duration_kind"] = pd.NA

    out.loc[
        duration_mask & out["period_days"].between(70, 120, inclusive="both"),
        "duration_kind",
    ] = "quarter"

    out.loc[
        duration_mask & out["period_days"].between(150, 210, inclusive="both"),
        "duration_kind",
    ] = "half_year"

    out.loc[
        duration_mask & out["period_days"].between(240, 300, inclusive="both"),
        "duration_kind",
    ] = "nine_month"

    out.loc[
        duration_mask & out["period_days"].between(320, 390, inclusive="both"),
        "duration_kind",
    ] = "annual"

    out.loc[
        duration_mask & out["duration_kind"].isna(),
        "duration_kind",
    ] = "other_duration"

    out["is_quarter"] = out["duration_kind"].eq("quarter")
    out["is_annual"] = out["duration_kind"].eq("annual")
    out["is_instant"] = out["reporting_kind"].eq("instant")

    return out


def latest_period_asof(
    observations: pd.DataFrame,
    *,
    decision_time: str | pd.Timestamp,
    ciks: set[int] | None = None,
    metrics: set[str] | None = None,
    duration_kind: str | None = None,
    reporting_kind: str | None = None,
) -> pd.DataFrame:
    """Return the latest economically distinct reporting period known by a decision time."""
    out = add_period_semantics(observations)

    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")

    out = out[out["information_time"] <= decision].copy()

    if ciks is not None:
        out = out[out["cik"].isin(ciks)].copy()

    if metrics is not None:
        out = out[out["metric"].isin(metrics)].copy()

    if duration_kind is not None:
        out = out[out["duration_kind"].eq(duration_kind)].copy()

    if reporting_kind is not None:
        out = out[out["reporting_kind"].eq(reporting_kind)].copy()

    if out.empty:
        return out

    period_identity = [
        c for c in [
            "cik",
            "metric",
            "unit",
            "reporting_kind",
            "duration_kind",
            "start",
            "end",
        ] if c in out.columns
    ]

    out = (
        out.sort_values(
            period_identity + ["information_time"],
            kind="mergesort",
        )
        .groupby(period_identity, dropna=False, as_index=False)
        .tail(1)
    )

    latest_period_end = (
        out.groupby(
            [c for c in ["cik", "metric", "unit"] if c in out.columns],
            dropna=False,
        )["end"]
        .transform("max")
    )

    out = out[out["end"].eq(latest_period_end)].copy()

    return out.sort_values(
        [
            c for c in [
                "cik",
                "metric",
                "unit",
                "reporting_kind",
                "duration_kind",
                "end",
                "information_time",
            ] if c in out.columns
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def build_semantic_feature_names(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Create stable model-facing SEC feature names while retaining source semantics."""
    out = add_period_semantics(observations)

    def make_name(row: pd.Series) -> str:
        metric = str(row["metric"])
        kind = str(row["reporting_kind"])

        if kind == "instant":
            return f"sec_{metric}_instant"

        duration = str(row["duration_kind"])
        return f"sec_{metric}_{duration}"

    out["feature_name"] = out.apply(make_name, axis=1)
    return out
