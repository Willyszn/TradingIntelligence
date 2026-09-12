
import pandas as pd

from trading_intelligence.research.unified_panel import (
    add_three_day_market_labels,
    build_sec_pit_features,
    build_unified_research_panel,
)


def _market():
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": "2024-01-02T21:00:00Z",
                "open": 100,
                "high": 102,
                "low": 99,
                "close": 101,
                "volume": 1000,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-01-03T21:00:00Z",
                "open": 102,
                "high": 104,
                "low": 101,
                "close": 103,
                "volume": 1100,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-01-04T21:00:00Z",
                "open": 104,
                "high": 105,
                "low": 103,
                "close": 104,
                "volume": 1200,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-01-05T21:00:00Z",
                "open": 105,
                "high": 107,
                "low": 104,
                "close": 106,
                "volume": 1300,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-01-08T21:00:00Z",
                "open": 107,
                "high": 108,
                "low": 106,
                "close": 107,
                "volume": 1400,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-01-09T21:00:00Z",
                "open": 108,
                "high": 109,
                "low": 107,
                "close": 108,
                "volume": 1500,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-01-10T21:00:00Z",
                "open": 109,
                "high": 110,
                "low": 108,
                "close": 109,
                "volume": 1600,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-10T21:00:00Z",
                "open": 110,
                "high": 111,
                "low": 109,
                "close": 110,
                "volume": 1700,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-11T21:00:00Z",
                "open": 111,
                "high": 112,
                "low": 110,
                "close": 111,
                "volume": 1800,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-12T21:00:00Z",
                "open": 112,
                "high": 113,
                "low": 111,
                "close": 112,
                "volume": 1900,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-13T21:00:00Z",
                "open": 113,
                "high": 114,
                "low": 112,
                "close": 113,
                "volume": 2000,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-15T21:00:00Z",
                "open": 110,
                "high": 111,
                "low": 109,
                "close": 110,
                "volume": 1700,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-16T21:00:00Z",
                "open": 111,
                "high": 112,
                "low": 110,
                "close": 111,
                "volume": 1800,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-20T21:00:00Z",
                "open": 112,
                "high": 113,
                "low": 111,
                "close": 112,
                "volume": 1900,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-21T21:00:00Z",
                "open": 113,
                "high": 114,
                "low": 112,
                "close": 113,
                "volume": 2000,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-02-22T21:00:00Z",
                "open": 114,
                "high": 115,
                "low": 113,
                "close": 114,
                "volume": 2100,
            },
        ]
    )


def _sec():
    rows = []

    quarters = [
        (
            "2023-01-01",
            "2023-03-31",
            100,
            "2023-05-01",
        ),
        (
            "2023-04-01",
            "2023-06-30",
            110,
            "2023-08-01",
        ),
        (
            "2023-07-01",
            "2023-09-30",
            120,
            "2023-11-01",
        ),
        (
            "2023-10-01",
            "2023-12-31",
            130,
            "2024-02-01",
        ),
    ]

    for start, end, value, info in quarters:
        rows.append(
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": start,
                "end": end,
                "val_num": value,
                "information_time": (
                    f"{info}T12:00:00Z"
                ),
            }
        )

    # Later revision to Q4.
    rows.append(
        {
            "cik": 1,
            "metric": "revenue",
            "unit": "USD",
            "start": "2023-10-01",
            "end": "2023-12-31",
            "val_num": 999,
            "information_time": (
                "2024-02-15T12:00:00Z"
            ),
        }
    )

    rows.append(
        {
            "cik": 1,
            "metric": "assets",
            "unit": "USD",
            "start": None,
            "end": "2023-12-31",
            "val_num": 1000,
            "information_time": (
                "2024-02-01T12:00:00Z"
            ),
        }
    )

    return pd.DataFrame(rows)


def test_three_day_labels_use_next_open_and_t3_close():
    out = add_three_day_market_labels(
        _market()
    )

    first = out.iloc[0]

    assert first["entry_price"] == 102
    assert first["exit_price"] == 106

    assert abs(
        first["target_return"]
        - (106 / 102 - 1)
    ) < 1e-12


def test_sec_features_are_point_in_time():
    decisions = pd.DataFrame(
        [
            {
                "_decision_id": 1,
                "symbol": "AAA",
                "decision_time": (
                    "2024-02-10T21:00:00Z"
                ),
                "cik": 1,
            }
        ]
    )

    out = build_sec_pit_features(
        decisions,
        _sec(),
    )

    row = out.iloc[0]

    # Q4 130 is known after Feb 1.
    assert row[
        "sec_revenue_quarter"
    ] == 130


def test_sec_revision_becomes_available_only_after_revision_time():
    decisions = pd.DataFrame(
        [
            {
                "_decision_id": 1,
                "symbol": "AAA",
                "decision_time": (
                    "2024-02-10T21:00:00Z"
                ),
                "cik": 1,
            },
            {
                "_decision_id": 2,
                "symbol": "AAA",
                "decision_time": (
                    "2024-02-20T21:00:00Z"
                ),
                "cik": 1,
            },
        ]
    )

    out = build_sec_pit_features(
        decisions,
        _sec(),
    )

    first = out[
        out["_decision_id"] == 1
    ].iloc[0]

    second = out[
        out["_decision_id"] == 2
    ].iloc[0]

    assert first[
        "sec_revenue_quarter"
    ] == 130

    assert second[
        "sec_revenue_quarter"
    ] == 999


def test_unified_panel_contains_market_sec_and_fred():
    master = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": 1,
            }
        ]
    )

    fred = pd.DataFrame(
        [
            {
                "series_id": "TEST",
                "date": "2023-12-01",
                "value": 50,
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-12-31",
                "available_date_conservative": (
                    "2024-01-02"
                ),
            }
        ]
    )

    out = build_unified_research_panel(
        _market(),
        sec_observations=_sec(),
        fred_observations=fred,
        security_master=master,
    )

    first = out.iloc[0]

    assert first["entry_price"] == 102
    assert first["exit_price"] == 106

    # First decision is Jan 2, before Q4 disclosure.
    assert first[
        "sec_revenue_quarter"
    ] == 120

    assert first["macro_TEST"] == 50


def test_unified_panel_applies_sec_revision_at_correct_time():
    master = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": 1,
            }
        ]
    )

    out = build_unified_research_panel(
        _market(),
        sec_observations=_sec(),
        security_master=master,
    )

    pre_revision = out[
        out["decision_time"]
        < pd.Timestamp(
            "2024-02-15T12:00:00Z"
        )
    ].copy()

    post_revision = out[
        out["decision_time"]
        >= pd.Timestamp(
            "2024-02-15T12:00:00Z"
        )
    ].copy()

    assert not pre_revision.empty
    assert not post_revision.empty

    assert (
        pre_revision.iloc[-1][
            "sec_revenue_quarter"
        ]
        == 130
    )

    assert (
        post_revision.iloc[0][
            "sec_revenue_quarter"
        ]
        == 999
    )


def test_no_future_information_is_used():
    master = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": 1,
            }
        ]
    )

    out = build_unified_research_panel(
        _market(),
        sec_observations=_sec(),
        security_master=master,
    )

    q4_revision_time = pd.Timestamp(
        "2024-02-15T12:00:00Z"
    )

    before = out[
        out["decision_time"]
        < q4_revision_time
    ]

    after = out[
        out["decision_time"]
        >= q4_revision_time
    ]

    assert (
        bool(
            before[
                "sec_revenue_quarter"
            ]
            .eq(999)
            .any()
        )
        is False
    )

    assert (
        after[
            "sec_revenue_quarter"
        ]
        .eq(999)
        .any()
    )
