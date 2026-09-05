from __future__ import annotations

import pandas as pd


def chronological_purged_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.7,
    purge_bars: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by unique timestamps and purge labels whose target ends in the test period."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if purge_bars < 0:
        raise ValueError("purge_bars must be >= 0")
    required = {"timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values")
    df = df.sort_values(["timestamp", "symbol"] if "symbol" in df.columns else ["timestamp"]).reset_index(drop=True)
    timestamps = pd.Index(df["timestamp"].drop_duplicates().sort_values())
    if len(timestamps) < 2:
        raise ValueError("At least 2 unique timestamps are required")
    cut_idx = max(1, min(len(timestamps) - 1, int(len(timestamps) * train_fraction)))
    test_start = pd.Timestamp(timestamps[cut_idx]).tz_convert("UTC")
    train = df[df["timestamp"] < test_start].copy()
    test = df[df["timestamp"] >= test_start].copy()

    if "target_timestamp" in train.columns:
        tt = pd.to_datetime(train["target_timestamp"], utc=True, errors="coerce")
        train = train[tt < test_start].copy()

    # Optional additional embargo after the last retained training timestamp.
    if purge_bars and not train.empty and "symbol" in df.columns:
        # Timestamp-based embargo: remove the last N unique training timestamps.
        train_ts = train["timestamp"].drop_duplicates().sort_values()
        if len(train_ts) > purge_bars:
            cutoff = train_ts.iloc[-purge_bars]
            train = train[train["timestamp"] < cutoff].copy()
        else:
            train = train.iloc[0:0].copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)
