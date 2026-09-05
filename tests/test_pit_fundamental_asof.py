import pandas as pd
import pytest

from trading_intelligence.research.pit_fundamental_asof import asof_latest_observations


def test_asof_never_uses_future_information():
    obs = pd.DataFrame([
        {"cik": 1, "metric": "revenue", "unit": "USD", "start": "2024-01-01", "end": "2024-03-31", "information_time": "2024-05-01T15:00:00Z", "val_num": 100, "period_type": "duration", "accn": "a1"},
        {"cik": 1, "metric": "revenue", "unit": "USD", "start": "2024-01-01", "end": "2024-03-31", "information_time": "2024-05-20T15:00:00Z", "val_num": 110, "period_type": "duration", "accn": "a2"},
    ])
    dec = pd.DataFrame([
        {"cik": 1, "decision_time": "2024-05-10T21:00:00Z"},
        {"cik": 1, "decision_time": "2024-05-25T21:00:00Z"},
    ])
    out = asof_latest_observations(obs, dec)
    assert out["val_num"].tolist() == [100, 110]
    assert (out["information_time"] <= out["decision_time"]).all()


def test_no_future_observation_returns_unavailable():
    obs = pd.DataFrame([
        {"cik": 1, "metric": "assets", "unit": "USD", "start": None, "end": "2024-03-31", "information_time": "2024-05-01T15:00:00Z", "val_num": 500, "period_type": "instant"},
    ])
    dec = pd.DataFrame([{"cik": 1, "decision_time": "2024-04-30T21:00:00Z"}])
    out = asof_latest_observations(obs, dec)
    assert len(out) == 1
    assert not bool(out.iloc[0]["available"])


def test_requires_decision_columns():
    with pytest.raises(ValueError):
        asof_latest_observations(pd.DataFrame(), pd.DataFrame({"cik": [1]}))
