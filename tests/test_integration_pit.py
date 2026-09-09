
import pandas as pd

from trading_intelligence.research.fred_pit import (
    align_fred_asof,
)
from trading_intelligence.research.sec_derived_features import (
    add_all_derived_features,
)
from trading_intelligence.research.sec_semantics import (
    add_period_semantics,
)
from trading_intelligence.research.unified_panel import (
    add_three_day_market_labels,
    build_sec_pit_features,
    build_unified_research_panel,
)


def _market():
    rows = []

    dates = pd.bdate_range(
        "2024-01-02",
        "2024-02-22",
    )

    for symbol in [
        "AAA",
        "BBB",
    ]:
        for i, date in enumerate(
            dates
        ):
            open_price = 100 + i
            close_price = 101 + i

            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": (
                        date.strftime(
                            "%Y-%m-%d"
                        )
                        + "T21:00:00Z"
                    ),
                    "open": open_price,
                    "high": open_price + 2,
                    "low": open_price - 1,
                    "close": close_price,
                    "volume": 1000 + i,
                }
            )

    return pd.DataFrame(rows)


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

    # Later revision to the SAME economic period.
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


def _fred():
    return pd.DataFrame(
        [
            {
                "series_id": "TEST_A",
                "date": "2023-12-01",
                "value": 50,
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-12-31",
                "available_date_conservative": (
                    "2024-01-02"
                ),
            },
            {
                "series_id": "TEST_A",
                "date": "2023-12-01",
                "value": 55,
                "realtime_start": "2024-02-01",
                "realtime_end": "2024-12-31",
                "available_date_conservative": (
                    "2024-02-02"
                ),
            },
            {
                "series_id": "TEST_B",
                "date": "2023-11-01",
                "value": 75,
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-12-31",
                "available_date_conservative": (
                    "2024-01-02"
                ),
            },
        ]
    )


def test_market_labels_are_session_based():
    out = add_three_day_market_labels(
        _market()
    )

    first = out[
        out["symbol"] == "AAA"
    ].iloc[0]

    assert first["entry_price"] == 102

    assert first["target_timestamp"] == pd.Timestamp(
        "2024-01-05T21:00:00Z"
    )

    assert first["exit_price"] == 104


def test_sec_derived_features_are_index_safe():
    source = _sec()

    source = source.iloc[
        [4, 0, 3, 1, 5, 2]
    ].copy()

    source = add_period_semantics(
        source
    )

    out = add_all_derived_features(
        source
    )

    q4 = out[
        (out["end"] == pd.Timestamp(
            "2023-12-31",
            tz="UTC",
        ))
        & (
            out["information_time"]
            == pd.Timestamp(
                "2024-02-01",
                tz="UTC",
            )
        )
    ]

    assert len(q4) == 1


def test_sec_revision_is_pit_safe():
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


def test_fred_future_vintage_does_not_mask_old_vintage():
    decisions = pd.DataFrame(
        [
            {
                "_decision_id": 1,
                "decision_time": (
                    "2024-02-01T21:00:00Z"
                ),
            },
            {
                "_decision_id": 2,
                "decision_time": (
                    "2024-02-02T21:00:00Z"
                ),
        ]
    )

    out = align_fred_asof(
        decisions,
        _fred(),
    )

    first = out[
        (out["_decision_id"] == 1)
        & (out["series_id"] == "TEST_A")
    ].iloc[0]

    second = out[
        (out["_decision_id"] == 2)
        & (out["series_id"] == "TEST_A")
    ].iloc[0]

    assert first["value"] == 50
    assert second["value"] == 55


def test_unified_panel_crosses_sec_revision_boundary():
    master = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": 1,
            }
        ]
    )

    market = _market().query(
        "symbol == 'AAA'"
    )

    out = build_unified_research_panel(
        market,
        sec_observations=_sec(),
        security_master=master,
    )

    before = out[
        out["decision_time"]
        < pd.Timestamp(
            "2024-02-15T12:00:00Z"
        )
    ]

    after = out[
        out["decision_time"]
        >= pd.Timestamp(
            "2024-02-15T12:00:00Z"
        )
    ]

    assert not before.empty
    assert not after.empty

    assert (
        before[
            "sec_revenue_quarter"
        ]
        .eq(999)
        .any()
        is False
    )

    assert (
        after[
            "sec_revenue_quarter"
        ]
        .eq(999)
        .any()
    )


def test_unified_panel_one_row_per_decision():
    master = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": 1,
            },
            {
                "symbol": "BBB",
                "cik": 1,
            },
        ]
    )

    out = build_unified_research_panel(
        _market(),
        sec_observations=_sec(),
        fred_observations=_fred(),
        security_master=master,
    )

    assert not out[
        ["symbol", "decision_time"]
    ].duplicated().any()

    assert "target_return" in out
    assert (
        "sec_revenue_quarter"
        in out
    )
    assert "macro_TEST_A" in out
    assert "macro_TEST_B" in out


def test_missing_cik_is_not_a_pipeline_error():
    out = build_unified_research_panel(
        _market().query(
            "symbol == 'AAA'"
        ),
        sec_observations=_sec(),
        security_master=None,
    )

    assert not out.empty
    assert "target_return" in out
    assert (
        "sec_revenue_quarter"
        not in out.columns
    )
