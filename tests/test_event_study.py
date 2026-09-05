import pandas as pd

from trading_intelligence.research.event_study import event_study


def test_event_study_uses_first_bar_after_event():
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 11:00", "2025-01-01 12:00"], utc=True),
        "symbol": ["XYZ"] * 3,
        "close": [100, 102, 104],
    })
    events = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 10:30"], utc=True),
        "asset": ["XYZ"],
        "event_type": ["news"],
        "sentiment": [0.8],
    })
    out = event_study(events, bars, horizons=(1,))
    assert abs(out.iloc[0]["forward_return_1"] - (104 / 102 - 1)) < 1e-9
