import pandas as pd

from trading_intelligence.research.sec_semantics import (
    add_period_semantics,
    build_semantic_feature_names,
    latest_period_asof,
)


def _obs():
    return pd.DataFrame(
        [
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2024-01-01",
                "end": "2024-03-31",
                "val_num": 100,
                "information_time": "2024-05-01T10:00:00Z",
            },
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2024-01-01",
                "end": "2024-03-31",
                "val_num": 110,
                "information_time": "2024-05-20T10:00:00Z",
            },
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2024-01-01",
                "end": "2024-06-30",
                "val_num": 220,
                "information_time": "2024-08-01T10:00:00Z",
            },
            {
                "cik": 1,
                "metric": "assets",
                "unit": "USD",
                "start": None,
                "end": "2024-06-30",
                "val_num": 1000,
                "information_time": "2024-08-01T10:00:00Z",
            },
        ]
    )


def test_period_semantics_classify_instant_and_quarter():
    out = add_period_semantics(_obs())

    revenue = out[out["metric"] == "revenue"]
    assets = out[out["metric"] == "assets"]

    assert revenue["reporting_kind"].eq("duration").all()
    assert revenue.iloc[0]["duration_kind"] == "quarter"
    assert assets.iloc[0]["reporting_kind"] == "instant"
    assert bool(assets.iloc[0]["is_instant"]) is True


def test_revision_does_not_change_period_identity():
    out = add_period_semantics(_obs())

    q1 = out[
        (out["metric"] == "revenue")
        & (out["end"] == pd.Timestamp("2024-03-31", tz="UTC"))
    ]

    assert len(q1) == 2
    assert q1["val_num"].tolist() == [100, 110]


def test_latest_period_asof_respects_information_time():
    out = latest_period_asof(
        _obs(),
        decision_time="2024-05-25T21:00:00Z",
        metrics={"revenue"},
    )

    assert len(out) == 1
    assert out.iloc[0]["val_num"] == 110


def test_future_revision_is_excluded():
    out = latest_period_asof(
        _obs(),
        decision_time="2024-05-10T21:00:00Z",
        metrics={"revenue"},
    )

    assert len(out) == 1
    assert out.iloc[0]["val_num"] == 100


def test_feature_names_are_stable():
    out = build_semantic_feature_names(_obs())

    names = set(out["feature_name"])

    assert "sec_revenue_quarter" in names
    assert "sec_assets_instant" in names
