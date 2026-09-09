
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

    out["symbol"] = out["symbol"].astype(str)

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
            f"SEC observations missing columns: {missing}"
        )

    out = observations.copy()

    out["cik"] = pd.to_numeric(
        out["cik"],
        errors="coerce",
    ).astype("Int64")

    out["information_time"] = pd.to_datetime(
        out["information_time"],
        utc=True,
        errors="coerce",
    )

    out["start"] = pd.to_datetime(
        out["start"],
        utc=True,
        errors="coerce",
    )

    out["end"] = pd.to_datetime(
        out["end"],
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
            "information_time",
            "end",
            "val_num",
        ]
    ).copy()

    return add_period_semantics(
        out
    ).reset_index(drop=True)


def _latest_revision_per_period(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "cik",
        "metric",
        "unit",
        "reporting_kind",
        "duration_kind",
        "start",
        "end",
    ]

    work = observations.sort_values(
        keys + ["information_time"],
        kind="mergesort",
    )

    return (
        work.groupby(
            keys,
            dropna=False,
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )


def _sec_features_for_one_decision(
    decision: pd.Series,
    observations: pd.DataFrame,
) -> dict:
    result = {
        "_decision_id": decision["_decision_id"]
    }

    cik = decision.get("cik")

    if pd.isna(cik):
        return result

    obs = observations[
        observations["cik"] == int(cik)
    ].copy()

    if obs.empty:
        return result

    obs = obs[
        obs["information_time"]
        <= decision["decision_time"]
    ].copy()

    if obs.empty:
        return result

    obs = _latest_revision_per_period(
        obs
    )

    for metric in sorted(
        obs["metric"].dropna().unique()
    ):
        metric_rows = obs[
            obs["metric"] == metric
        ].copy()

        if metric_rows.empty:
            continue

        # ----------------------------------------------------
        # Latest known instant fact
        # ----------------------------------------------------
        instant = metric_rows[
            metric_rows["reporting_kind"]
            == "instant"
        ].copy()

        if not instant.empty:
            instant = instant.sort_values(
                [
                    "end",
                    "information_time",
                ],
                kind="mergesort",
            )

            row = instant.iloc[-1]

            result[
                f"sec_{metric}_instant"
            ] = float(row["val_num"])

            result[
                f"sec_{metric}_instant_age_days"
            ] = (
                decision["decision_time"]
                - row["information_time"]
            ).total_seconds() / 86400.0

        # ----------------------------------------------------
        # Latest known quarterly duration
        # ----------------------------------------------------
        quarters = metric_rows[
            metric_rows["duration_kind"]
            == "quarter"
        ].copy()

        if quarters.empty:
            continue

        quarters = quarters.sort_values(
            [
                "end",
                "information_time",
            ],
            kind="mergesort",
        )

        latest_q = quarters.iloc[-1]

        result[
            f"sec_{metric}_quarter"
        ] = float(latest_q["val_num"])

        result[
            f"sec_{metric}_quarter_age_days"
        ] = (
            decision["decision_time"]
            - latest_q["information_time"]
        ).total_seconds() / 86400.0

        # Derived features are calculated AFTER PIT filtering.
        derived = add_all_derived_features(
            metric_rows
        )

        latest_rows = derived[
            (
                derived["start"]
                == latest_q["start"]
            )
            & (
                derived["end"]
                == latest_q["end"]
            )
            & (
                derived["information_time"]
                == latest_q["information_time"]
            )
        ]

        if latest_rows.empty:
            continue

        latest = latest_rows.iloc[-1]

        if pd.notna(latest["qoq_growth"]):
            result[
                f"sec_{metric}_qoq_growth"
            ] = float(
                latest["qoq_growth"]
            )

        if pd.notna(latest["yoy_growth"]):
            result[
                f"sec_{metric}_yoy_growth"
            ] = float(
                latest["yoy_growth"]
            )

        if pd.notna(latest["ttm_value"]):
            result[
                f"sec_{metric}_ttm"
            ] = float(
                latest["ttm_value"]
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

    features = [
        _sec_features_for_one_decision(
            row,
            obs,
        )
        for _, row in dec.iterrows()
    ]

    if not features:
        return pd.DataFrame(
            columns=["_decision_id"]
        )

    return pd.DataFrame(
        features
    )


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
        dropna=False,
    ):
        row = {
            "_decision_id": decision_id
        }

        for _, item in group.iterrows():
            if not bool(item["available"]):
                continue

            series_id = str(
                item["series_id"]
            )

            row[
                f"macro_{series_id}"
            ] = float(item["value"])

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

    return pd.DataFrame(
        rows
    )


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

    out["symbol"] = out["symbol"].astype(str)

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        utc=True,
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
            "Market data contains duplicate "
            "symbol/timestamp observations."
        )

    out = out.sort_values(
        [
            "symbol",
            "timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    out["decision_time"] = out["timestamp"]

    grouped = out.groupby(
        "symbol",
        sort=False,
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

    out["forward_positive"] = (
        out["target_return"] > 0
    )

    out["label_horizon_sessions"] = 3
    out["decision_to_entry_sessions"] = 1

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
        master_required = {
            "symbol",
            "cik",
        }

        missing = sorted(
            master_required
            - set(security_master.columns)
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
            .drop_duplicates(
                "symbol"
            )
            .copy()
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
            suffixes=("", "_master"),
        )

        if "cik_master" in panel.columns:
            if "cik" not in panel.columns:
                panel["cik"] = panel[
                    "cik_master"
                ]
            else:
                panel["cik"] = panel[
                    "cik"
                ].fillna(
                    panel["cik_master"]
                )

            panel = panel.drop(
                columns=["cik_master"]
            )

    # --------------------------------------------------------
    # SEC PIT
    # --------------------------------------------------------
    if sec_observations is not None:
        sec_features = (
            build_sec_pit_features(
                panel[
                    [
                        "_decision_id",
                        "symbol",
                        "decision_time",
                        "cik",
                    ]
                ],
                sec_observations,
            )
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
    # Universal PIT assertions
    # --------------------------------------------------------
    if "information_time" in panel.columns:
        information_time = pd.to_datetime(
            panel["information_time"],
            utc=True,
            errors="coerce",
        )

        if (
            information_time.notna()
            & (
                information_time
                > panel["decision_time"]
            )
        ).any():
            raise AssertionError(
                "SEC information timestamp after "
                "decision timestamp."
            )

    return panel.drop(
        columns=["_decision_id"],
        errors="ignore",
    )
