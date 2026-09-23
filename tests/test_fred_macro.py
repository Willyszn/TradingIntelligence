import pandas as pd

from trading_intelligence.research.fred_macro import (
    DEFAULT_SERIES,
    _parse_vintage_observations,
    audit_macro_vintages,
    fetch_series_vintages,
)


def test_default_macro_set_is_present_and_named():
    assert len(DEFAULT_SERIES) == 9
    assert DEFAULT_SERIES["EFFR"] == "effective_federal_funds_rate"
    assert DEFAULT_SERIES["VIXCLS"] == "vix"


def test_macro_audit_passes_clean_dataset():
    df = pd.DataFrame(
        [
            {
                "series_id": "EFFR",
                "observation_date": "2024-01-01",
                "value": 5.3,
                "realtime_start": "2024-01-02",
                "realtime_end": "2024-01-02",
            },
            {
                "series_id": "DGS10",
                "observation_date": "2024-01-02",
                "value": 4.1,
                "realtime_start": "2024-01-03",
                "realtime_end": "2024-01-03",
            },
        ]
    )

    result = audit_macro_vintages(df)

    assert result["status"] == "PASS"
    assert result["duplicate_rows"] == 0


def test_macro_audit_detects_inverted_realtime_window():
    df = pd.DataFrame(
        [
            {
                "series_id": "EFFR",
                "observation_date": "2024-01-01",
                "value": 5.3,
                "realtime_start": "2024-02-02",
                "realtime_end": "2024-02-01",
            },
        ]
    )

    result = audit_macro_vintages(df)

    assert result["status"] == "FAIL"
    assert result["realtime_inversion_rows"] == 1


def test_vintage_parser_converts_fred_wide_output():
    observations = [
        {
            "date": "2024-01-01",
            "EFFR_20240105": "5.00",
            "EFFR_20240205": "5.10",
        }
    ]

    out = _parse_vintage_observations(
        "EFFR",
        observations,
    )

    assert len(out) == 2

    frame = pd.DataFrame(out).sort_values(
        "realtime_start"
    )

    assert frame["value"].tolist() == [5.0, 5.1]
    assert frame["realtime_start"].dt.tz is not None


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "observations": [
                {
                    "date": "2024-01-01",
                    "EFFR_20240105": "5.0",
                    "EFFR_20240205": "5.1",
                },
                {
                    "date": "2024-01-02",
                    "EFFR_20240105": "5.1",
                    "EFFR_20240205": "5.2",
                },
            ]
        }


class _FakeSession:
    def __init__(self):
        self.params = []

    def get(self, url, params, timeout):
        self.params.append(params)
        return _FakeResponse()

    def close(self):
        return None


def test_fetch_series_vintages_requests_and_parses_wide_vintages():
    session = _FakeSession()

    out = fetch_series_vintages(
        "EFFR",
        "test-key",
        observation_start="2022-01-01",
        observation_end="2026-09-02",
        vintage_start="2023-01-01",
        vintage_end="2026-09-02",
        session=session,
    )

    assert len(out) == 4
    assert out["realtime_start"].nunique() == 2

    params = session.params[0]

    assert params["output_type"] == 2
    assert params["realtime_start"] == "2023-01-01"
    assert params["realtime_end"] == "2026-09-02"
