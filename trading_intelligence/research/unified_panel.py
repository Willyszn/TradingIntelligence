
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_intelligence.research.sec_derived_features import (
    add_all_derived_features,
)
from trading_intelligence.research.sec_semantics import (
    add_period_semantics,
)


def _normalise_decisions(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "symbol",
        "decision_time",
    }

    missing = sorted(
        required - set(decisions.columns)
    )

    if missing:
        raise ValueError(
            f"Decision frame missing columns: {missing}"
        )

    out = decisions.copy()

    out["symbol"] = (
        out["symbol"].astype(str)
    )

    out["decision_time"] = pd.to_datetime(
        out["decision_time"],
        utc=True,
        errors="coerce",
    )

    if "cik" in out.columns:
        out["cik"] = pd.to_numeric(
            out["cik"],
            errors="coerce",
        ).astype("Int64")

    out = out.dropna(
        subset=[
            "symbol",
            "decision_time",
        ]
    ).copy()

    if "_decision_id" not in out.columns:
        out["_decision_id"] = np.arange(
            len(out),
            dtype=np.int64,
        )

    if out["_decision_id"].duplicated().any():
        raise ValueError(
            "_decision_id must be unique."
        )

    return out.reset_index(drop=True)


def _prepare_sec(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "cik",
        "metric",
        "unit",
        "start",
        "end",
        "information_time",
        "val_num",
    }

    missing = sorted(
        required - set(observations.columns)
    )

    if missing:
        raise ValueError(
            "SEC observations missing columns: "
            f"{missing}"
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

    out = out.dropna(
        subset=[
            "cik",
            "metric",
            "unit",
            "end",
            "information_time",
            "val_num",
        ]
    ).copy()

    return (
        add_period_semantics(out)
        .reset_index(drop=True)
    )


def _pit_snapshot(
    observations: pd.DataFrame,
    decision_time: pd.Timestamp,
) -> pd.DataFrame:
    # CRITICAL:
    # filter by information_time FIRST.
    # No globally latest revision is allowed to leak backward.
    known = observations[
        observations["information_time"]
        <= decision_time
    ].copy()

    if known.empty:
        return known

    keys = [
        "cik",
        "metric",
        "unit",
        "reporting_kind",
        "duration_kind",
        "start",
        "end",
    ]

    return (
        known.sort_values(
            keys + ["information_time"],
            kind="mergesort",
        )
        .groupby(
            keys,
            dropna=False,
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )


def _build_one_sec_snapshot(
    decision: pd.Series,
    observations: pd.DataFrame,
) -> dict:
    result = {
        "_decision_id": decision[
            "_decision_id"
        ]
    }

    cik = decision.get("cik")

    if pd.isna(cik):
        return result

    company_obs = observations[
        observations["cik"] == int(cik)
    ].copy()

    if company_obs.empty:
        return result

    snapshot = _pit_snapshot(
        company_obs,
        decision["decision_time"],
    )

    if snapshot.empty:
        return result

    for metric in sorted(
        snapshot["metric"]
        .dropna()
        .unique()
    ):
        metric_rows = snapshot[
            snapshot["metric"] == metric
        ].copy()

        # -----------------------------
        # Latest instant fact
        # -----------------------------
        instant = metric_rows[
            metric_rows["reporting_kind"]
            == "instant"
        ].copy()

        if not instant.empty:
            instant = instant.sort_values(
                "end",
                kind="mergesort",
            )

            row = instant.iloc[-1]

            result[
                f"sec_{metric}_instant"
            ] = float(
                row["val_num"]
            )

            result[
                f"sec_{metric}_instant_age_days"
            ] = (
                decision["decision_time"]
                - row["information_time"]
            ).total_seconds() / 86400.0

        # -----------------------------
        # Latest quarterly observation
        # -----------------------------
        quarters = metric_rows[
            metric_rows["duration_kind"]
            == "quarter"
        ].copy()

        if quarters.empty:
            continue

        quarters = quarters.sort_values(
            "end",
            kind="mergesort",
        )

        latest = quarters.iloc[-1]

        result[
            f"sec_{metric}_quarter"
        ] = float(
            latest["val_num"]
        )

        result[
            f"sec_{metric}_quarter_age_days"
        ] = (
            decision["decision_time"]
            - latest["information_time"]
        ).total_seconds() / 86400.0

        # Derived features operate on the PIT
        # snapshot, never on today's revised history.
        derived = add_all_derived_features(
            metric_rows
        )

        latest_derived = derived[
            (
                derived["start"]
                == latest["start"]
            )
            & (
                derived["end"]
                == latest["end"]
            )
            & (
                derived["information_time"]
                == latest["information_time"]
            )
        ]

        if latest_derived.empty:
            continue

        row = latest_derived.iloc[-1]

        if pd.notna(
            row["qoq_growth"]
        ):
            result[
                f"sec_{metric}_qoq_growth"
            ] = float(
                row["qoq_growth"]
            )

        if pd.notna(
            row["yoy_growth"]
        ):
            result[
                f"sec_{metric}_yoy_growth"
            ] = float(
                row["yoy_growth"]
            )

        if pd.notna(
            row["ttm_value"]
        ):
            result[
                f"sec_{metric}_ttm"
            ] = float(
                row["ttm_value"]
            )

    return result


def build_sec_pit_features(
    decisions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    dec = _normalise_decisions(
        decisions
    )

    obs = _prepare_sec(
        observations
    )

    rows = [
        _build_one_sec_snapshot(
            decision,
            obs,
        )
        for _, decision in dec.iterrows()
    ]

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return pd.DataFrame(
            columns=["_decision_id"]
        )

    if result[
        "_decision_id"
    ].duplicated().any():
        raise AssertionError(
            "SEC feature builder returned "
            "duplicate decision IDs."
        )

    return result


def build_fred_pit_features(
    decisions: pd.DataFrame,
    fred_observations: pd.DataFrame,
) -> pd.DataFrame:
    from trading_intelligence.research.fred_pit import (
        align_fred_asof,
    )

    dec = _normalise_decisions(
        decisions
    )

    aligned = align_fred_asof(
        dec[
            [
                "_decision_id",
                "symbol",
                "decision_time",
            ]
        ],
        fred_observations,
    )

    if aligned.empty:
        return pd.DataFrame(
            columns=["_decision_id"]
        )

    rows = []

    for decision_id, group in aligned.groupby(
        "_decision_id",
        sort=False,
    ):
        row = {
            "_decision_id": decision_id
        }

        for _, item in group.iterrows():
            if not bool(
                item["available"]
            ):
                continue

            series_id = str(
                item["series_id"]
            )

            row[
                f"macro_{series_id}"
            ] = float(
                item["value"]
            )

            if pd.notna(
                item["observation_date"]
            ):
                decision_date = (
                    pd.Timestamp(
                        item["decision_time"]
                    )
                    .tz_convert(None)
                    .normalize()
                )

                observation_date = (
                    pd.Timestamp(
                        item["observation_date"]
                    )
                    .normalize()
                )

                row[
                    f"macro_{series_id}_age_days"
                ] = (
                    decision_date
                    - observation_date
                ).days

        rows.append(row)

    result = pd.DataFrame(rows)

    if result[
        "_decision_id"
    ].duplicated().any():
        raise AssertionError(
            "FRED feature builder returned "
            "duplicate decision IDs."
        )

    return result


def add_three_day_market_labels(
    market: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = sorted(
        required - set(market.columns)
    )

    if missing:
        raise ValueError(
            f"Market data missing columns: {missing}"
        )

    out = market.copy()

    out["symbol"] = (
        out["symbol"].astype(str)
    )

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    out = out.dropna(
        subset=[
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ).copy()

    if out.duplicated(
        ["symbol", "timestamp"]
    ).any():
        raise ValueError(
            "Duplicate market "
            "symbol/timestamp rows."
        )

    out = out.sort_values(
        [
            "symbol",
            "timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = out.groupby(
        "symbol",
        sort=False,
    )

    out["decision_time"] = (
        out["timestamp"]
    )

    out["entry_timestamp"] = (
        grouped["timestamp"].shift(-1)
    )

    out["entry_price"] = (
        grouped["open"].shift(-1)
    )

    out["target_timestamp"] = (
        grouped["timestamp"].shift(-3)
    )

    out["exit_price"] = (
        grouped["close"].shift(-3)
    )

    out["target_return"] = (
        out["exit_price"]
        / out["entry_price"]
        - 1.0
    )

    out["forward_return"] = (
        out["target_return"]
    )

    out["forward_positive"] = (
        out["target_return"] > 0
    ).astype("Int64")

    out["label_horizon_sessions"] = 3
    out["decision_to_entry_sessions"] = 1

    invalid_entry = (
        out["entry_timestamp"].notna()
        & (
            out["entry_timestamp"]
            <= out["decision_time"]
        )
    )

    if invalid_entry.any():
        raise AssertionError(
            "Market entry timestamp is not "
            "strictly after decision timestamp."
        )

    return out


def build_unified_research_panel(
    market: pd.DataFrame,
    sec_observations: pd.DataFrame | None = None,
    fred_observations: pd.DataFrame | None = None,
    security_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    panel = add_three_day_market_labels(
        market
    )

    panel = panel.dropna(
        subset=[
            "entry_price",
            "exit_price",
            "target_timestamp",
        ]
    ).copy()

    panel = panel.reset_index(
        drop=True
    )

    panel["_decision_id"] = np.arange(
        len(panel),
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Security master
    # --------------------------------------------------------
    if security_master is not None:
        required = {
            "symbol",
            "cik",
        }

        missing = sorted(
            required - set(
                security_master.columns
            )
        )

        if missing:
            raise ValueError(
                "Security master missing columns: "
                f"{missing}"
            )

        master = (
            security_master[
                ["symbol", "cik"]
            ]
            .copy()
            .drop_duplicates(
                "symbol"
            )
        )

        master["symbol"] = (
            master["symbol"].astype(str)
        )

        master["cik"] = pd.to_numeric(
            master["cik"],
            errors="coerce",
        ).astype("Int64")

        panel = panel.merge(
            master,
            on="symbol",
            how="left",
            suffixes=(
                "",
                "_master",
            ),
            validate="many_to_one",
        )

        if "cik_master" in panel.columns:
            if "cik" not in panel.columns:
                panel["cik"] = (
                    panel["cik_master"]
                )
            else:
                panel["cik"] = (
                    panel["cik"]
                    .fillna(
                        panel["cik_master"]
                    )
                )

            panel = panel.drop(
                columns=["cik_master"]
            )

    # Make missing CIK an unavailable SEC feature,
    # not a pipeline error.
    if sec_observations is not None:
        sec_decisions = panel[
            [
                "_decision_id",
                "symbol",
                "decision_time",
            ]
        ].copy()

        if "cik" in panel.columns:
            sec_decisions["cik"] = (
                panel["cik"]
            )

        else:
            sec_decisions["cik"] = (
                pd.Series(
                    pd.array(
                        [pd.NA] * len(panel),
                        dtype="Int64",
                    ),
                    index=panel.index,
                )
            )

        sec_features = build_sec_pit_features(
            sec_decisions,
            sec_observations,
        )

        panel = panel.merge(
            sec_features,
            on="_decision_id",
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # FRED PIT
    # --------------------------------------------------------
    if fred_observations is not None:
        fred_features = (
            build_fred_pit_features(
                panel[
                    [
                        "_decision_id",
                        "symbol",
                        "decision_time",
                    ]
                ],
                fred_observations,
            )
        )

        panel = panel.merge(
            fred_features,
            on="_decision_id",
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # Final invariants
    # --------------------------------------------------------
    if panel[
        "_decision_id"
    ].duplicated().any():
        raise AssertionError(
            "Unified panel contains duplicate "
            "decision IDs."
        )

    if len(panel) != panel[
        "_decision_id"
    ].nunique():
        raise AssertionError(
            "Unified panel violates the "
            "one-row-per-decision invariant."
        )

    if (
        panel["entry_timestamp"]
        <= panel["decision_time"]
    ).any():
        raise AssertionError(
            "Entry occurs at or before decision time."
        )

    return panel.drop(
        columns=["_decision_id"],
        errors="ignore",
    )
