from __future__ import annotations

from pathlib import Path
import pandas as pd

# Explicit raw-tag precedence. A higher number wins when multiple raw tags represent
# the same research metric and the same CIK/period/unit/information timestamp.
TAG_PRECEDENCE = {
    # Prefer current US-GAAP revenue tag over legacy Revenues.
    'RevenueFromContractWithCustomerExcludingAssessedTax': 20,
    'Revenues': 10,
    'NetIncomeLoss': 20,
    'Assets': 20,
    'Liabilities': 20,
    # Prefer StockholdersEquity as the common-shareholders equity concept when both
    # tags are present. The NCI-inclusive tag remains available as a separate lineage.
    'StockholdersEquity': 20,
    'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest': 10,
    'CashAndCashEquivalentsAtCarryingValue': 20,
    'LongTermDebtAndFinanceLeaseObligationsCurrent': 30,
    'LongTermDebtCurrent': 30,
    'LongTermDebtNoncurrent': 30,
    'LongTermDebt': 10,
    'EarningsPerShareDiluted': 20,
}

METRIC_MAP = {
    'RevenueFromContractWithCustomerExcludingAssessedTax': 'revenue',
    'Revenues': 'revenue',
    'NetIncomeLoss': 'net_income',
    'Assets': 'assets',
    'Liabilities': 'liabilities',
    'StockholdersEquity': 'equity',
    'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest': 'equity_including_nci',
    'CashAndCashEquivalentsAtCarryingValue': 'cash',
    'LongTermDebtAndFinanceLeaseObligationsCurrent': 'debt_current',
    'LongTermDebtCurrent': 'debt_current',
    'LongTermDebtNoncurrent': 'debt_noncurrent',
    'LongTermDebt': 'debt_total',
    'EarningsPerShareDiluted': 'eps_diluted',
}

REQUIRED_COLUMNS = ['cik', 'entity_name', 'taxonomy', 'tag', 'label', 'description', 'unit',
                    'start', 'end', 'filed', 'form', 'frame', 'fy', 'fp', 'accn', 'val',
                    'information_time', 'metric']


def _ensure_metric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'metric' not in out.columns or out['metric'].isna().all():
        out['metric'] = out['tag'].map(METRIC_MAP)
    else:
        out['metric'] = out['metric'].fillna(out['tag'].map(METRIC_MAP))
    return out[out['metric'].notna()].copy()


def _period_type(df: pd.DataFrame) -> pd.Series:
    start = pd.to_datetime(df['start'], utc=True, errors='coerce')
    end = pd.to_datetime(df['end'], utc=True, errors='coerce')
    days = (end - start).dt.days
    return days.where(start.notna(), 0).pipe(lambda s: s.apply(lambda x: 'instant' if x == 0 else ('annual' if 300 <= x <= 380 else 'duration')))


def resolve_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Resolve same-timestamp metric aliases without deleting raw economic history.

    The result is one canonical observation per CIK/metric/unit/start/end/information_time.
    When aliases collide, the highest explicit TAG_PRECEDENCE wins; ties break by raw tag.
    Later filings are *not* collapsed here—information_time remains part of the key so
    every revision is preserved for point-in-time as-of queries.
    """
    work = _ensure_metric(df)
    if work.empty:
        return work, {'input_rows': 0, 'output_rows': 0, 'alias_collisions': 0, 'exact_source_duplicates': 0}

    for col in ['information_time', 'filed', 'start', 'end']:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], utc=True, errors='coerce')
    work['val_num'] = pd.to_numeric(work['val'], errors='coerce')
    work = work[work['val_num'].notna()].copy()
    work['tag_precedence'] = work['tag'].map(TAG_PRECEDENCE).fillna(0).astype(int)
    work['period_type'] = _period_type(work)

    key = ['cik', 'metric', 'unit', 'start', 'end', 'information_time']
    dup_mask = work.duplicated(key, keep=False)
    alias_collisions = int(dup_mask.sum())

    # Stable deterministic winner; raw SEC tag and accession remain in lineage.
    work = work.sort_values(key + ['tag_precedence', 'tag', 'accn'], kind='mergesort')
    resolved = work.drop_duplicates(key, keep='last').copy()
    resolved = resolved.sort_values(['cik', 'metric', 'period_type', 'end', 'information_time', 'accn'], kind='mergesort').reset_index(drop=True)

    # Count exact source duplicates against the full filing identity.
    source_key = ['cik', 'tag', 'unit', 'start', 'end', 'information_time', 'accn', 'val_num']
    exact_source = int(work.duplicated(source_key, keep=False).sum())

    stats = {
        'input_rows': int(len(df)),
        'eligible_rows': int(len(work)),
        'output_rows': int(len(resolved)),
        'rows_reduced': int(len(work) - len(resolved)),
        'alias_collision_rows': alias_collisions,
        'exact_source_duplicate_rows': exact_source,
        'unique_ciks': int(resolved['cik'].nunique()),
    }
    return resolved, stats


def write_observations(input_csv: str | Path, output_csv: str | Path, *, chunksize: int = 250_000) -> dict:
    """Build a compact canonical PIT observation file from the streaming PIT events."""
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # The file is only ~550k rows in the current universe, but we keep chunking in the
    # interface so the layer scales with the universe later. We aggregate chunks and a
    # final pass because the same key can cross chunk boundaries.
    frames: list[pd.DataFrame] = []
    rows = 0
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        rows += len(chunk)
        frames.append(chunk)
    if not frames:
        pd.DataFrame(columns=REQUIRED_COLUMNS + ['val_num', 'tag_precedence', 'period_type']).to_csv(output_csv, index=False)
        return {'input_rows': 0, 'output_rows': 0, 'output': str(output_csv)}

    all_df = pd.concat(frames, ignore_index=True)
    resolved, stats = resolve_observations(all_df)
    # Keep canonical fields plus lineage useful for audit/debugging.
    cols = [c for c in REQUIRED_COLUMNS + ['val_num', 'tag_precedence', 'period_type'] if c in resolved.columns]
    for extra in ['reportDate']:
        if extra in resolved.columns and extra not in cols:
            cols.append(extra)
    resolved[cols].to_csv(output_csv, index=False)
    stats.update({'output': str(output_csv), 'input_rows_read': rows})
    return stats
