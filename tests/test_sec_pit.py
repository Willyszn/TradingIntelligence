import pandas as pd
import pytest

from trading_intelligence.research.sec_pit import (
    align_sec_asof,
    build_sec_feature_panel,
    load_canonical_sec_observations,
)


def _observations():
    return pd.DataFrame(
        [
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2024-01-01",
                "end": "2024-03-31",
                "period_type": "duration",
                "information_time": "2024-05-01T15:00:00Z",
                "val_num": 100,
                "accn": "a1",
            },
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2024-01-01",
                "end": "2024-03-31",
                "period_type": "duration",
                "information_time": "2024-05-20T15:00:00Z",
                "val_num": 110,
                "accn": "a2",
            },
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2024-01-01",
                "end": "2024-06-30",
                "period_type": "duration",
                "information_time": "2024-08-01T15:00:00Z",
                "val_num": 210,
                "accn": "a3",
            },
        ]
    )


def test_sec_asof_uses_latest_known_revision():
    decisions = pd.DataFrame(
        [{"cik": 1, "decision_time": "2024-05-25T21:00:00Z"}]
    )

    out = align_sec_asof(decisions, _observations())

    revenue = out[(out["metric"] == "revenue") & out["available"]]
    assert len(revenue) == 1
    assert revenue.iloc[0]["val_num"] == 110
    assert revenue.iloc[0]["information_time"] == pd.Timestamp(
        "2024-05-20T15:00:00Z"
    )


def test_sec_asof_preserves_distinct_economic_periods():
    decisions = pd.DataFrame(
        [{"cik": 1, "decision_time": "2024-08-10T21:00:00Z"}]
    )

    out = align_sec_asof(decisions, _observations())

    assert len(out) == 2
    assert set(out["end"].dt.strftime("%Y-%m-%d")) == {
        "2024-03-31",
        "2024-06-30",
    }


def test_future_sec_information_is_never_used():
    decisions = pd.DataFrame(
        [{"cik": 1, "decision_time": "2024-04-30T21:00:00Z"}]
    )

    out = align_sec_asof(decisions, _observations())

    assert not bool(out["available"].any())


def test_feature_panel_has_observation_age():
    decisions = pd.DataFrame(
        [{"cik": 1, "decision_time": "2024-05-25T21:00:00Z"}]
    )

    out = build_sec_feature_panel(decisions, _observations())

    assert bool(out.iloc[0]["feature_available"]) is True
    assert out.iloc[0]["observation_age_days"] > 4


def test_loader_filters_ciks_and_metrics(tmp_path):
    source = tmp_path / "sec.csv"
    _observations().to_csv(source, index=False)

    out = load_canonical_sec_observations(
        source,
        ciks={1},
        metrics={"revenue"},
    )

    assert len(out) == 3


def test_requires_core_sec_columns():
    with pytest.raises(ValueError):
        load_canonical_sec_observations(
            pd.DataFrame({"cik": [1]}),
        )
