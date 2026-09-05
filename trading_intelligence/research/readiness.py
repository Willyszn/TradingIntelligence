from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetReadiness:
    rows: int
    symbols: int
    min_rows_per_symbol: int
    median_rows_per_symbol: float
    max_rows_per_symbol: int
    start: str
    end: str
    calendar_days: int
    warnings: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_market_dataset(
    frame: pd.DataFrame,
    *,
    min_rows_per_symbol: int = 750,
    min_symbols: int = 5,
    require_multiple_years: bool = True,
) -> DatasetReadiness:
    """Assess whether a normalized market dataset is large enough for ML research.

    This is a gate, not a claim of model validity. Passing means the dataset is
    large enough to *start* controlled research; it does not imply predictive edge.
    """
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol"])
    counts = df.groupby("symbol").size()
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    calendar_days = int((end - start).days) if pd.notna(start) and pd.notna(end) else 0

    warnings: list[str] = []
    if len(df) == 0:
        warnings.append("Dataset is empty.")
    if len(counts) < min_symbols:
        warnings.append(f"Only {len(counts)} symbols found; target at least {min_symbols} for multi-asset research.")
    if counts.min() < min_rows_per_symbol:
        warnings.append(
            f"At least one symbol has only {int(counts.min())} rows; target at least {min_rows_per_symbol} per symbol before ML conclusions."
        )
    if require_multiple_years and calendar_days < 365 * 2:
        warnings.append("Historical coverage is under two calendar years; regime robustness will be limited.")

    if warnings:
        status = "NOT_READY" if len(df) == 0 or len(counts) < min_symbols or counts.min() < min_rows_per_symbol else "READY_WITH_WARNINGS"
    else:
        status = "READY"

    return DatasetReadiness(
        rows=int(len(df)),
        symbols=int(len(counts)),
        min_rows_per_symbol=int(counts.min()) if not counts.empty else 0,
        median_rows_per_symbol=float(counts.median()) if not counts.empty else 0.0,
        max_rows_per_symbol=int(counts.max()) if not counts.empty else 0,
        start=start.isoformat() if pd.notna(start) else "",
        end=end.isoformat() if pd.notna(end) else "",
        calendar_days=calendar_days,
        warnings=tuple(warnings),
        status=status,
    )
