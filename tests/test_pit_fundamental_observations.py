import pandas as pd
from trading_intelligence.research.pit_fundamental_observations import resolve_observations


def base(tag, metric_time='2024-02-01T10:00:00Z', val=100):
    return {
        'cik': 1, 'entity_name': 'X', 'taxonomy': 'us-gaap', 'tag': tag,
        'label': '', 'description': '', 'unit': 'USD', 'start': '2023-01-01',
        'end': '2023-12-31', 'filed': '2024-02-01', 'form': '10-K', 'frame': 'CY2023',
        'fy': 2023, 'fp': 'FY', 'accn': 'a'+tag[:2], 'val': val,
        'information_time': metric_time, 'metric': None,
    }


def test_alias_precedence_reduces_same_timestamp_collision():
    df = pd.DataFrame([base('Revenues', val=90), base('RevenueFromContractWithCustomerExcludingAssessedTax', val=100)])
    out, stats = resolve_observations(df)
    assert len(out) == 1
    assert float(out.iloc[0]['val_num']) == 100
    assert stats['rows_reduced'] == 1


def test_revisions_at_different_information_times_are_preserved():
    a = base('Assets', metric_time='2024-01-01T10:00:00Z', val=100)
    b = base('Assets', metric_time='2024-03-01T10:00:00Z', val=110)
    out, stats = resolve_observations(pd.DataFrame([a, b]))
    assert len(out) == 2
    assert stats['rows_reduced'] == 0


def test_debt_components_are_not_collapsed_into_total_debt():
    a = base('LongTermDebtCurrent', val=20)
    b = base('LongTermDebtNoncurrent', val=80)
    c = base('LongTermDebt', val=100)
    out, _ = resolve_observations(pd.DataFrame([a, b, c]))
    assert set(out['metric']) == {'debt_current', 'debt_noncurrent', 'debt_total'}
