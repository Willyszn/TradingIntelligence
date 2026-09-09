import pandas as pd

from trading_intelligence.research.sec_semantics import add_period_semantics
from trading_intelligence.research.sec_derived_features import (
    add_all_derived_features,
    add_qoq_growth,
    add_ttm,
    add_yoy_growth,
)


def _quarter(cik, metric, unit, start, end, value, info):
    return {
        "cik": cik,
        "metric": metric,
        "unit": unit,
        "start": start,
        "end": end,
        "val_num": value,
        "information_time": info,
    }


def _observations():
    rows = [
        _quarter(1, "revenue", "USD", "2023-01-01", "2023-03-31", 100, "2023-05-01T10:00:00Z"),
        _quarter(1, "revenue", "USD", "2023-04-01", "2023-06-30", 110, "2023-08-01T10:00:00Z"),
        _quarter(1, "revenue", "USD", "2023-07-01", "2023-09-30", 120, "2023-11-01T10:00:00Z"),
        _quarter(1, "revenue", "USD", "2023-10-01", "2023-12-31", 130, "2024-02-01T10:00:00Z"),
        _quarter(1, "revenue", "USD", "2024-01-01", "2024-03-31", 150, "2024-05-01T10:00:00Z"),
        _quarter(1, "revenue", "USD", "2024-04-01", "2024-06-30", 165, "2024-08-01T10:00:00Z"),
    ]

    frame = pd.DataFrame(rows)
    return add_period_semantics(frame)


def test_qoq_growth():
    out = add_qoq_growth(_observations())

    row = out[out["end"] == pd.Timestamp("2024-06-30", tz="UTC")].iloc[0]

    assert abs(row["qoq_growth"] - 0.10) < 1e-12


def test_yoy_growth():
    out = add_yoy_growth(_observations())

    row = out[out["end"] == pd.Timestamp("2024-03-31", tz="UTC")].iloc[0]

    assert abs(row["yoy_growth"] - 0.50) < 1e-12


def test_ttm_uses_four_known_quarters():
    out = add_ttm(_observations())

    row = out[out["end"] == pd.Timestamp("2023-12-31", tz="UTC")].iloc[0]

    assert row["ttm_value"] == 460.0


def test_ttm_is_not_available_before_four_quarters():
    out = add_ttm(_observations())

    row = out[out["end"] == pd.Timestamp("2023-09-30", tz="UTC")].iloc[0]

    assert pd.isna(row["ttm_value"])


def test_future_quarter_does_not_enter_ttm():
    frame = _observations()

    # Make the 2024-Q2 quarter information available only after Q1's disclosure.
    frame.loc[
        frame["end"] == pd.Timestamp("2024-06-30", tz="UTC"),
        "information_time",
    ] = pd.Timestamp("2024-08-01T10:00:00Z")

    out = add_ttm(frame)

    q1 = out[out["end"] == pd.Timestamp("2024-03-31", tz="UTC")].iloc[0]

    # Q2 is not allowed into Q1's information set.
    assert q1["ttm_value"] == 510.0


def test_combined_features_preserve_source_rows():
    source = _observations()
    out = add_all_derived_features(source)

    assert len(out) == len(source)
    assert "qoq_growth" in out.columns
    assert "yoy_growth" in out.columns
    assert "ttm_value" in out.columns
    assert bool(out["ttm_available"].any()) is True
