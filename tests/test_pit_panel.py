import pandas as pd
import pytest

from trading_intelligence.research.pit_panel import (
    align_point_in_time,
    assert_no_future_information,
    build_three_day_labels,
)
from trading_intelligence.markets.security_master import (
    SecurityRecord,
    build_security_master,
    validate_security_master,
)


def test_pit_alignment_allows_exact_information_time():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "decision_time": ["2025-01-10T00:00:00Z"],
        }
    )
    observations = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "information_time": ["2025-01-10T00:00:00Z"],
            "period_end": ["2024-12-31"],
            "value": [123.0],
        }
    )

    aligned, result = align_point_in_time(
        decisions,
        observations,
        period_keys=("period_end",),
        value_columns=("value",),
    )

    assert result.matched_rows == 1
    assert aligned.loc[0, "pit_value"] == 123.0


def test_pit_alignment_excludes_future_information():
    decisions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "decision_time": ["2025-01-10T00:00:00Z"],
        }
    )
    observations = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "information_time": [
                "2025-01-09T00:00:00Z",
                "2025-01-11T00:00:00Z",
            ],
            "period_end": ["2024-12-31", "2024-12-31"],
            "value": [100.0, 200.0],
        }
    )

    aligned, _ = align_point_in_time(
        decisions,
        observations,
        period_keys=("period_end",),
        value_columns=("value",),
    )

    assert aligned.loc[0, "pit_value"] == 100.0
    assert_no_future_information(aligned)


def test_next_open_to_t_plus_3_close():
    market = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "timestamp": pd.date_range(
                "2025-01-01", periods=5, freq="D", tz="UTC"
            ),
            "open": [10, 11, 12, 13, 14],
            "close": [10, 11, 12, 13, 14],
        }
    )

    out = build_three_day_labels(market)

    assert out.loc[0, "entry_price"] == 11
    assert out.loc[0, "target_timestamp"] == out.loc[3, "timestamp"]
    assert out.loc[0, "forward_return_3d"] == pytest.approx(13 / 11 - 1)


def test_missing_future_bar_makes_label_unavailable():
    market = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 3,
            "timestamp": pd.date_range(
                "2025-01-01", periods=3, freq="D", tz="UTC"
            ),
            "open": [10, 11, 12],
            "close": [10, 11, 12],
        }
    )

    out = build_three_day_labels(market)

    assert bool(out.loc[2, "label_available"]) is False


def test_leveraged_etf_is_not_plain_common_equity():
    frame = build_security_master(
        [
            SecurityRecord(
                symbol="SPY",
                security_type="etf",
                is_etf=True,
            ),
            SecurityRecord(
                symbol="TQQQ",
                security_type="leveraged_etf",
                is_etf=True,
                is_leveraged=True,
            ),
        ]
    )

    result = validate_security_master(frame)
    assert result["status"] == "PASS"
    assert frame.loc[frame["symbol"] == "TQQQ", "is_common_equity"].item() is False
