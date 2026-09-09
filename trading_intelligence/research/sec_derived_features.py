
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


def _normalise(
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
        "duration_kind",
        "reporting_kind",
    }

    missing = sorted(
        required - set(observations.columns)
    )

    if missing:
        raise ValueError(
            f"SEC observations missing columns: {missing}"
        )

    out = observations.copy()

    # Stable source-row identity.
    if "__sec_source_id" not in out.columns:
        out["__sec_source_id"] = np.arange(
            len(out),
            dtype=np.int64,
        )

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

    return out.reset_index(
        drop=True
    )


def _latest_revision_per_period(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Return the latest revision per economic period.

    This is used only as a calculation view. It never replaces the
    caller's full observation frame.
    """
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
        .reset_index(drop=True)
    )


def _quarter_adjacent(
    current_end: pd.Timestamp,
    previous_end: pd.Timestamp,
) -> bool:
    if pd.isna(
        current_end
    ) or pd.isna(
        previous_end
    ):
        return False

    gap = (
        current_end
        - previous_end
    ).days

    return 70 <= gap <= 110


def _year_apart(
    current_end: pd.Timestamp,
    previous_end: pd.Timestamp,
) -> bool:
    if pd.isna(
        current_end
    ) or pd.isna(
        previous_end
    ):
        return False

    gap = (
        current_end
        - previous_end
    ).days

    return 300 <= gap <= 430


def _write_by_source_id(
    out: pd.DataFrame,
    source_id: int,
    column: str,
    value: float,
) -> None:
    out.loc[
        out["__sec_source_id"]
        == source_id,
        column,
    ] = value


def add_qoq_growth(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = _normalise(
        observations
    )

    out["qoq_growth"] = np.nan

    for _, group in out.groupby(
        [
            "cik",
            "metric",
            "unit",
        ],
        dropna=False,
        sort=False,
    ):
        quarters = group[
            group["duration_kind"]
            == "quarter"
        ].copy()

        if quarters.empty:
            continue

        latest = _latest_revision_per_period(
            quarters
        )

        latest = latest.sort_values(
            "end",
            kind="mergesort",
        ).reset_index(drop=True)

        for position in range(
            1,
            len(latest),
        ):
            current = latest.iloc[position]
            previous = latest.iloc[
                position - 1
            ]

            if not _quarter_adjacent(
                current["end"],
                previous["end"],
            ):
                continue

            denominator = previous[
                "val_num"
            ]

            if (
                pd.isna(denominator)
                or denominator == 0
            ):
                continue

            growth = (
                current["val_num"]
                / denominator
                - 1.0
            )

            _write_by_source_id(
                out,
                int(
                    current[
                        "__sec_source_id"
                    ]
                ),
                "qoq_growth",
                float(growth),
            )

    return out.reset_index(
        drop=True
    )


def add_yoy_growth(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = _normalise(
        observations
    )

    out["yoy_growth"] = np.nan

    for _, group in out.groupby(
        [
            "cik",
            "metric",
            "unit",
        ],
        dropna=False,
        sort=False,
    ):
        quarters = group[
            group["duration_kind"]
            == "quarter"
        ].copy()

        if len(quarters) < 5:
            continue

        latest = _latest_revision_per_period(
            quarters
        )

        latest = latest.sort_values(
            "end",
            kind="mergesort",
        ).reset_index(drop=True)

        for position in range(
            4,
            len(latest),
        ):
            current = latest.iloc[position]
            previous = latest.iloc[
                position - 4
            ]

            if not _year_apart(
                current["end"],
                previous["end"],
            ):
                continue

            denominator = previous[
                "val_num"
            ]

            if (
                pd.isna(denominator)
                or denominator == 0
            ):
                continue

            growth = (
                current["val_num"]
                / denominator
                - 1.0
            )

            _write_by_source_id(
                out,
                int(
                    current[
                        "__sec_source_id"
                    ]
                ),
                "yoy_growth",
                float(growth),
            )

    return out.reset_index(
        drop=True
    )


def add_ttm(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = _normalise(
        observations
    )

    out["ttm_value"] = np.nan

    for _, group in out.groupby(
        [
            "cik",
            "metric",
            "unit",
        ],
        dropna=False,
        sort=False,
    ):
        quarters = group[
            group["duration_kind"]
            == "quarter"
        ].copy()

        if len(quarters) < 4:
            continue

        latest = _latest_revision_per_period(
            quarters
        )

        latest = latest.sort_values(
            "end",
            kind="mergesort",
        ).reset_index(drop=True)

        for position in range(
            3,
            len(latest),
        ):
            window = latest.iloc[
                position - 3 : position + 1
            ]

            ends = window[
                "end"
            ].tolist()

            if not all(
                _quarter_adjacent(
                    ends[i],
                    ends[i - 1],
                )
                for i in range(
                    1,
                    len(ends),
                )
            ):
                continue

            current = latest.iloc[
                position
            ]

            value = float(
                window["val_num"].sum()
            )

            _write_by_source_id(
                out,
                int(
                    current[
                        "__sec_source_id"
                    ]
                ),
                "ttm_value",
                value,
            )

    out["ttm_available"] = (
        out["ttm_value"].notna()
    )

    return out.reset_index(
        drop=True
    )


def add_all_derived_features(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived features without collapsing source observations."""
    out = add_qoq_growth(
        observations
    )

    out = add_yoy_growth(
        out
    )

    out = add_ttm(
        out
    )

    # Internal lineage key is not a model feature.
    return out.drop(
        columns=[
            "__sec_source_id"
        ],
        errors="ignore",
    ).reset_index(
        drop=True
    )
