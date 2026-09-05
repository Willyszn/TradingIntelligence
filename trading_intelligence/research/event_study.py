from __future__ import annotations

import pandas as pd


def event_study(events: pd.DataFrame, bars: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """Measure forward returns after point-in-time events, using the first bar at/after the event."""
    required_e = {"timestamp", "asset", "event_type", "sentiment"}
    required_b = {"timestamp", "symbol", "close"}
    if not required_e.issubset(events.columns):
        raise ValueError(f"Events missing columns: {sorted(required_e - set(events.columns))}")
    if not required_b.issubset(bars.columns):
        raise ValueError(f"Bars missing columns: {sorted(required_b - set(bars.columns))}")
    e = events.copy()
    b = bars.copy()
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b = b.sort_values(["symbol", "timestamp"])
    rows: list[dict] = []
    for _, event in e.sort_values("timestamp").iterrows():
        series = b[b["symbol"] == event["asset"]]
        if series.empty:
            continue
        after = series[series["timestamp"] >= event["timestamp"]]
        if after.empty:
            continue
        base = after.iloc[0]
        ordered = series.reset_index(drop=True)
        match = ordered.index[ordered["timestamp"] == base["timestamp"]]
        if len(match) == 0:
            continue
        i = int(match[0])
        rec = {
            "event_timestamp": event["timestamp"],
            "asset": event["asset"],
            "event_type": event["event_type"],
            "sentiment": event["sentiment"],
            "base_timestamp": base["timestamp"],
        }
        for h in horizons:
            j = i + int(h)
            rec[f"forward_return_{h}"] = (float(ordered.iloc[j]["close"]) / float(base["close"]) - 1.0) if j < len(ordered) else None
        rows.append(rec)
    return pd.DataFrame(rows)
