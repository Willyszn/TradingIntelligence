
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
        start="2024-01-02",
        end="2024-02-29",
    )

    for symbol in (
        "AAA",
        "BBB",
    ):
        for i, date in enumerate(
            dates
        ):
            open_price = 100.0 + i
            close_price = 101.0 + i

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
                    "high": open_price + 2.0,
                    "low": open_price - 1.0,
                    "close": close_price,
                    "volume": 1000.0 + i,
                }
            )

    return pd.DataFrame(
        rows
    )


def _sec():
    return pd.DataFrame(
        [
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2023-01-01",
                "end": "2023-03-31",
                "val_num": 100.0,
                "information_time": "2023-05-01T12:00:00Z",
            },
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2023-04-01",
                "end": "2023-06-30",
                "val_num": 110.0,
                "information_time": "2023-08-01T12:00:00Z",
            },
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2023-07-01",
                "end": "2023-09-30",
                "val_num": 120.0,
                "information_time": "2023-11-01T12:00:00Z",
            },
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2023-10-01",
                "end": "2023-12-31",
                "val_num": 130.0,
                "information_time": "2024-02-01T12:00:00Z",
            },
            {
                "cik": 1,
                "metric": "revenue",
                "unit": "USD",
                "start": "2023-10-01",
                "end": "2023-12-31",
                "val_num": 999.0,
                "information_time": "2024-02-15T12:00:00Z",
            },
            {
                "cik": 1,
                "metric": "assets",
                "unit": "USD",
                "start": None,
                "end": "2023-12-31",
                "val_num": 1000.0,
                "information_time": "2024-02-01T12:00:00Z",
            },
        ]
    )


def _fred():
    return pd.DataFrame(
        [
            {
                "series_id": "TEST_A",
                "date": "2023-12-01",
                "value": 50.0,
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-12-31",
                "available_date_conservative": "2024-01-02",
            },
            {
                "series_id": "TEST_A",
                "date": "2023-12-01",
                "value": 55.0,
                "realtime_start": "2024-02-01",
                "realtime_end": "2024-12-31",
                "available_date_conservative": "2024-02-02",
            },
            {
                "series_id": "TEST_B",
                "date": "2023-11-01",
                "value": 75.0,
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-12-31",
                "available_date_conservative": "2024-01-02",
            },
        ]
    )


def test_market_label_uses_next_open_and_t3_close():
    out = add_three_day_market_labels(
        _market()
    )

    first = out[
        out["symbol"] == "AAA"
    ].iloc[0]

    # The fixture defines Jan 3 open as 101.
    assert first["entry_timestamp"] == pd.Timestamp(
        "2024-01-03T21:00:00Z"
    )

    assert first["entry_price"] == 101.0

    assert first["target_timestamp"] == pd.Timestamp(
        "2024-01-05T21:00:00Z"
    )

    # Jan 5 close = 104.
    assert first["exit_price"] == 104.0


def test_sec_derived_features_preserve_both_revisions():
    source = add_period_semantics(
        _sec().iloc[
            [4, 0, 3, 1, 5, 2]
        ].copy()
    )

    out = add_all_derived_features(
        source
    )

    q4_initial = out[
        (out["end"] == pd.Timestamp(
            "2023-12-31",
            tz="UTC",
        ))
        & (
            out["information_time"]
            == pd.Timestamp(
                "2024-02-01T12:00:00Z",
            )
        )
    ]

    q4_revision = out[
        (out["end"] == pd.Timestamp(
            "2023-12-31",
            tz="UTC",
        ))
        & (
            out["information_time"]
            == pd.Timestamp(
                "2024-02-15T12:00:00Z",
            )
        )
    ]

    assert len(q4_initial) == 1
    assert len(q4_revision) == 1

    assert q4_initial.iloc[0]["val_num"] == 130.0
    assert q4_revision.iloc[0]["val_num"] == 999.0


def test_sec_revision_is_selected_by_decision_time():
    decisions = pd.DataFrame(
        [
            {
                "_decision_id": 1,
                "symbol": "AAA",
                "decision_time": "2024-02-10T21:00:00Z",
                "cik": 1,
            },
            {
                "_decision_id": 2,
                "symbol": "AAA",
                "decision_time": "2024-02-20T21:00:00Z",
                "cik": 1,
            },
        ]
    )

    out = build_sec_pit_features(
        decisions,
        _sec(),
    )

    before = out[
        out["_decision_id"] == 1
    ].iloc[0]

    after = out[
        out["_decision_id"] == 2
    ].iloc[0]

    assert before[
        "sec_revenue_quarter"
    ] == 130.0

    assert after[
        "sec_revenue_quarter"
    ] == 999.0


def test_fred_revision_boundary():
    decisions = pd.DataFrame(
        [
            {
                "_decision_id": 1,
                "decision_time": "2024-02-01T21:00:00Z",
            },
            {
                "_decision_id": 2,
                "decision_time": "2024-02-02T21:00:00Z",
            },
        ]
    )

    out = align_fred_asof(
        decisions,
        _fred(),
    )

    before = out[
        (out["_decision_id"] == 1)
        & (out["series_id"] == "TEST_A")
    ].iloc[0]

    after = out[
        (out["_decision_id"] == 2)
        & (out["series_id"] == "TEST_A")
    ].iloc[0]

    assert before["value"] == 50.0
    assert after["value"] == 55.0


def test_unified_panel_crosses_sec_revision_boundary():
    master = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": 1,
            }
        ]
    )

    out = build_unified_research_panel(
        _market().query(
            "symbol == 'AAA'"
        ),
        sec_observations=_sec(),
        security_master=master,
    )

    revision_time = pd.Timestamp(
        "2024-02-15T12:00:00Z"
    )

    before = out[
        out["decision_time"]
        < revision_time
    ]

    after = out[
        out["decision_time"]
        >= revision_time
    ]

    assert not before.empty
    assert not after.empty

    assert not before[
        "sec_revenue_quarter"
    ].eq(999.0).any()

    assert after[
        "sec_revenue_quarter"
    ].eq(999.0).any()


def test_unified_panel_is_one_row_per_decision():
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
        [
            "symbol",
            "decision_time",
        ]
    ].duplicated().any()

    assert "target_return" in out.columns
    assert (
        "sec_revenue_quarter"
        in out.columns
    )
    assert "macro_TEST_A" in out.columns
    assert "macro_TEST_B" in out.columns


def test_missing_cik_is_not_a_pipeline_error():
    out = build_unified_research_panel(
        _market().query(
            "symbol == 'AAA'"
        ),
        sec_observations=_sec(),
        security_master=None,
    )

    assert not out.empty
    assert "target_return" in out.columns
    assert (
        "sec_revenue_quarter"
        not in out.columns
    )
