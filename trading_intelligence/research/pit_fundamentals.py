from __future__ import annotations

from pathlib import Path
import pandas as pd

DEFAULT_TAGS = {
    'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues'],
    'net_income': ['NetIncomeLoss'],
    'assets': ['Assets'],
    'liabilities': ['Liabilities'],
    'equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'cash': ['CashAndCashEquivalentsAtCarryingValue'],
    'debt': ['LongTermDebtAndFinanceLeaseObligationsCurrent', 'LongTermDebtCurrent', 'LongTermDebtNoncurrent', 'LongTermDebt'],
    'eps_diluted': ['EarningsPerShareDiluted'],
}


def _coerce_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')


def attach_information_time(companyfacts: pd.DataFrame, submissions: pd.DataFrame) -> pd.DataFrame:
    """Attach SEC acceptance timestamp by accession number.

    Company Facts carries the accession number (`accn`) and a filed date. The
    submissions table carries the more precise acceptance timestamp. We use
    acceptance time as the information-availability timestamp when present,
    falling back to the filed date at midnight UTC only when no accession match
    exists.
    """
    cf = companyfacts.copy()
    sub = submissions.copy()
    _coerce_datetime(cf, 'filed')
    _coerce_datetime(sub, 'acceptanceDateTime')
    sub['accn'] = sub.get('accessionNumber')
    keep = [c for c in ['cik', 'accn', 'acceptanceDateTime', 'filingDate', 'form', 'reportDate'] if c in sub.columns]
    if 'accn' not in keep or 'acceptanceDateTime' not in keep:
        raise ValueError('submissions must contain accessionNumber and acceptanceDateTime')
    sub = sub[keep].drop_duplicates(['cik', 'accn'], keep='last')
    out = cf.merge(sub, on=['cik', 'accn'], how='left', suffixes=('', '_submission'))
    out['information_time'] = out['acceptanceDateTime']
    missing = out['information_time'].isna() & out['filed'].notna()
    out.loc[missing, 'information_time'] = out.loc[missing, 'filed']
    return out


def latest_asof(
    facts: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    ciks: set[int] | None = None,
    tags: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Return the latest disclosed observation for each logical metric as of `asof`.

    This is a research utility, not a semantic reconciliation engine. It keeps
    the latest disclosed observation for each CIK/metric/unit/period, using the
    SEC information timestamp, and does not overwrite raw facts.
    """
    if 'information_time' not in facts.columns:
        raise ValueError('facts must include information_time; call attach_information_time first')
    tag_map = tags or DEFAULT_TAGS
    inverse = {raw: metric for metric, raws in tag_map.items() for raw in raws}
    work = facts.copy()
    work['metric'] = work['tag'].map(inverse)
    work = work[work['metric'].notna()].copy()
    if ciks is not None:
        work = work[work['cik'].astype('Int64').isin(list(ciks))]
    work['information_time'] = pd.to_datetime(work['information_time'], utc=True, errors='coerce')
    asof_ts = pd.Timestamp(asof)
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize('UTC')
    else:
        asof_ts = asof_ts.tz_convert('UTC')
    work = work[work['information_time'].notna() & (work['information_time'] <= asof_ts)]
    work['val_num'] = pd.to_numeric(work['val'], errors='coerce')
    work = work[work['val_num'].notna()]
    keys = ['cik', 'metric', 'unit', 'start', 'end']
    for k in keys:
        if k not in work.columns:
            work[k] = None
    work = work.sort_values(['cik', 'metric', 'unit', 'start', 'end', 'information_time', 'accn'])
    return work.groupby(keys, dropna=False, as_index=False).tail(1).reset_index(drop=True)


def make_fundamental_snapshot(latest: pd.DataFrame) -> pd.DataFrame:
    """Collapse point-in-time SEC facts into one row per CIK with basic metrics."""
    if latest.empty:
        return latest.copy()
    df = latest.copy()
    df['period_end'] = pd.to_datetime(df['end'], utc=True, errors='coerce')
    duration = df['start'].notna() & df['end'].notna()
    df['days'] = (pd.to_datetime(df['end'], utc=True, errors='coerce') - pd.to_datetime(df['start'], utc=True, errors='coerce')).dt.days
    annual = df[(~duration) | (df['days'].isna()) | (df['days'].between(300, 380))].copy()
    pivot = (annual.sort_values(['cik','metric','period_end','information_time'])
             .groupby(['cik','metric'], as_index=False)
             .tail(1)
             .pivot(index='cik', columns='metric', values='val_num')
             .reset_index())
    pivot.columns.name = None
    if 'revenue' in pivot.columns and 'net_income' in pivot.columns:
        pivot['net_margin'] = pivot['net_income'] / pivot['revenue'].replace(0, pd.NA)
    if 'assets' in pivot.columns and 'liabilities' in pivot.columns:
        pivot['debt_to_assets_proxy'] = pivot['liabilities'] / pivot['assets'].replace(0, pd.NA)
    return pivot
