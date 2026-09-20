import pandas as pd

from trading_intelligence.research.fred_pit import (
    align_fred_asof,
    latest_vintage_asof,
    load_fred_pit,
)


def _fred():
    return pd.DataFrame(
        [
            {
                "series_id": "TEST",
                "date": "2024-04-01",
                "value": 100,
                "realtime_start": "2024-04-10",
                "realtime_end": "2024-05-10",
                "available_date_conservative": "2024-04-11",
            },
            {
                "series_id": "TEST",
                "date": "2024-04-01",
                "value": 101,
                "realtime_start": "2024-05-10",
                "realtime_end": "2024-06-10",
                "available_date_conservative": "2024-05-11",
            },
            {
                "series_id": "TEST",
                "date": "2024-05-01",
                "value": 110,
                "realtime_start": "2024-05-11",
                "realtime_end": "2024-06-11",
                "available_date_conservative": "2024-05-12",
            },
        ]
    )


def test_loader_preserves_vintages():
    out = load_fred_pit(_fred())

    assert len(out) == 3
    assert out["date"].dt.tz is not None
    assert out["realtime_start"].dt.tz is not None
    assert out["available_date_conservative"].dt.tz is not None


def test_latest_vintage_asof_excludes_later_revision():
    out = latest_vintage_asof(
        _fred(),
        decision_time="2024-05-05T21:00:00Z",
    )

    row = out[
        out["date"]
        == pd.Timestamp("2024-04-01", tz="UTC")
    ].iloc[0]

    assert row["value"] == 100


def test_latest_vintage_asof_uses_new_revision_when_available():
    out = latest_vintage_asof(
        _fred(),
        decision_time="2024-05-20T21:00:00Z",
    )

    row = out[
        out["date"]
        == pd.Timestamp("2024-04-01", tz="UTC")
    ].iloc[0]

    assert row["value"] == 101


def test_alignment_accepts_utc_aware_decisions():
    decisions = pd.DataFrame(
        [
            {
                "decision_time": pd.Timestamp(
                    "2024-05-12T21:00:00Z"
                )
            },
        ]
    )

    out = align_fred_asof(
        decisions,
        _fred(),
    )

    assert len(out) == 1
    assert bool(out.iloc[0]["available"]) is True
    assert out.iloc[0]["value"] == 110


def test_alignment_excludes_future_vintage():
    decisions = pd.DataFrame(
        [
            {
                "decision_time": "2024-05-05T21:00:00Z"
            },
            {
                "decision_time": "2024-05-12T21:00:00Z"
            },
        ]
    )

    out = align_fred_asof(
        decisions,
        _fred(),
    )

    first = out.iloc[0]
    second = out.iloc[1]

    assert bool(first["available"]) is True
    assert first["value"] == 100

    assert bool(second["available"]) is True
    assert second["value"] == 110


def test_same_day_conservative_boundary():
    decisions = pd.DataFrame(
        [
            {
                "decision_time": "2024-05-10T23:00:00Z"
            },
            {
                "decision_time": "2024-05-11T23:00:00Z"
            },
        ]
    )

    out = align_fred_asof(
        decisions,
        _fred(),
    )

    first = out.iloc[0]
    second = out.iloc[1]

    assert first["value"] == 100
    assert second["value"] == 101


def test_actual_persisted_schema_observation_date():
    source = _fred().rename(
        columns={"date": "observation_date"}
    )

    out = load_fred_pit(source)

    assert "date" in out.columns
    assert out["date"].dt.tz is not None
