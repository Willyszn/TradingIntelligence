from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PITAlignmentResult:
    rows: int
    matched_rows: int
    unmatched_rows: int


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def align_point_in_time(
    decisions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    decision_key: str = "symbol",
    observation_key: str = "symbol",
    decision_time: str = "decision_time",
    information_time: str = "information_time",
    period_keys: tuple[str, ...] = (),
    value_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, PITAlignmentResult]:
    """As-of align observations using only information available by decision time.

    The latest observation satisfying information_time <= decision_time is selected.
    Economic period columns, when supplied, remain part of the matching identity so
    quarterly/annual or otherwise distinct periods are not collapsed together.
    """
    _require_columns(
        decisions,
        {decision_key, decision_time},
        "decisions",
    )
    _require_columns(
        observations,
        {observation_key, information_time, *period_keys},
        "observations",
    )

    left = decisions.copy().reset_index(drop=False).rename(columns={"index": "_row_id"})
    right = observations.copy()

    left[decision_time] = pd.to_datetime(
        left[decision_time], utc=True, errors="coerce"
    )
    right[information_time] = pd.to_datetime(
        right[information_time], utc=True, errors="coerce"
    )

    left = left.dropna(subset=[decision_key, decision_time]).copy()
    right = right.dropna(subset=[observation_key, information_time]).copy()

    right = right.sort_values(
        [observation_key, information_time, *period_keys],
        kind="mergesort",
    )
    left = left.sort_values(
        [decision_key, decision_time],
        kind="mergesort",
    )

    use_columns = list(value_columns) if value_columns is not None else [
        c
        for c in right.columns
        if c not in {observation_key, information_time, *period_keys}
    ]

    right_payload = right[
        [observation_key, information_time, *period_keys, *use_columns]
    ].copy()

    # Give the observation payload unique names so source timing is explicit.
    rename_map = {
        column: f"pit_{column}"
        for column in [information_time, *period_keys, *use_columns]
    }
    right_payload = right_payload.rename(columns=rename_map)

    # This first generic aligner matches by security and information time.
    # Economic-period columns are retained in the payload but are not used as
    # join keys here because decision rows do not carry a period identity.
    # SEC-specific period-preserving alignment will be handled by a dedicated
    # adapter in the next research-layer increment.
    aligned = pd.merge_asof(
        left.sort_values([decision_time, decision_key], kind="mergesort"),
        right_payload.sort_values(
            [f"pit_{information_time}", observation_key],
            kind="mergesort",
        ),
        left_on=decision_time,
        right_on=f"pit_{information_time}",
        left_by=decision_key,
        right_by=observation_key,
        direction="backward",
        allow_exact_matches=True,
    )

    aligned = aligned.sort_values("_row_id", kind="mergesort").reset_index(drop=True)

    matched_mask = aligned[f"pit_{information_time}"].notna()
    result = PITAlignmentResult(
        rows=int(len(aligned)),
        matched_rows=int(matched_mask.sum()),
        unmatched_rows=int((~matched_mask).sum()),
    )

    return aligned.drop(columns=["_row_id"]), result


def build_three_day_labels(
    market: pd.DataFrame,
    *,
    symbol_column: str = "symbol",
    timestamp_column: str = "timestamp",
    open_column: str = "open",
    close_column: str = "close",
    horizon: int = 3,
) -> pd.DataFrame:
    """Apply the project's fixed next-open entry / t+3 close exit convention."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    _require_columns(
        market,
        {
            symbol_column,
            timestamp_column,
            open_column,
            close_column,
        },
        "market",
    )

    df = market.copy()
    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column], utc=True, errors="coerce"
    )
    df = df.sort_values(
        [symbol_column, timestamp_column],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = df.groupby(symbol_column, group_keys=False)

    df["entry_timestamp"] = grouped[timestamp_column].shift(-1)
    df["entry_price"] = grouped[open_column].shift(-1)
    df["target_timestamp"] = grouped[timestamp_column].shift(-horizon)

    exit_price = grouped[close_column].shift(-horizon)
    df["forward_return_3d"] = exit_price / df["entry_price"] - 1.0

    positive = df["forward_return_3d"].notna()
    df["positive_3d_return"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )
    df.loc[positive, "positive_3d_return"] = (
        df.loc[positive, "forward_return_3d"] > 0
    ).astype("int8")

    # A signal row is research-valid only when both entry and exit exist.
    df["label_available"] = (
        df["entry_price"].notna()
        & df["target_timestamp"].notna()
        & df["forward_return_3d"].notna()
    )

    return df


def assert_no_future_information(
    aligned: pd.DataFrame,
    *,
    decision_time: str = "decision_time",
    pit_information_time: str = "pit_information_time",
) -> None:
    _require_columns(
        aligned,
        {decision_time, pit_information_time},
        "aligned",
    )

    decision = pd.to_datetime(
        aligned[decision_time], utc=True, errors="coerce"
    )
    information = pd.to_datetime(
        aligned[pit_information_time], utc=True, errors="coerce"
    )

    bad = information.notna() & (information > decision)
    if bad.any():
        raise AssertionError(
            f"PIT leakage detected in {int(bad.sum())} row(s)."
        )
