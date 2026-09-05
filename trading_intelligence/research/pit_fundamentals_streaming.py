from __future__ import annotations

from pathlib import Path
import csv
import json
import sqlite3
import zipfile
from collections import defaultdict

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

FACT_FIELDS = ['cik','entity_name','taxonomy','tag','label','description','unit','start','end','filed','form','frame','fy','fp','accn','val','information_time']


def metric_map(tags: dict[str, list[str]] | None = None) -> dict[str, str]:
    src = tags or DEFAULT_TAGS
    return {raw: metric for metric, raws in src.items() for raw in raws}


def build_acceptance_index(submissions_csv: str | Path, out_db: str | Path, *, ciks: set[int] | None = None, chunksize: int = 250_000) -> dict:
    """Build a compact SQLite accession -> acceptance timestamp index for selected CIKs."""
    submissions_csv = Path(submissions_csv)
    out_db = Path(out_db)
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()
    con = sqlite3.connect(out_db)
    try:
        con.execute('CREATE TABLE acceptance (cik INTEGER NOT NULL, accn TEXT NOT NULL, acceptance TEXT, filingDate TEXT, form TEXT, PRIMARY KEY(cik, accn))')
        con.execute('CREATE INDEX idx_acceptance_accn ON acceptance(accn)')
        total_rows = 0
        matched = 0
        usecols = ['cik','accessionNumber','acceptanceDateTime','filingDate','form']
        for chunk in pd.read_csv(submissions_csv, usecols=usecols, chunksize=chunksize, low_memory=False, dtype={'cik':'Int64','accessionNumber':'string','acceptanceDateTime':'string','filingDate':'string','form':'string'}):
            total_rows += len(chunk)
            if ciks is not None:
                chunk = chunk[chunk['cik'].isin(list(ciks))]
            if chunk.empty:
                continue
            chunk = chunk.dropna(subset=['cik','accessionNumber']).copy()
            if chunk.empty:
                continue
            rows = [tuple(x) for x in chunk[['cik','accessionNumber','acceptanceDateTime','filingDate','form']].itertuples(index=False, name=None)]
            con.executemany('INSERT OR REPLACE INTO acceptance(cik,accn,acceptance,filingDate,form) VALUES (?,?,?,?,?)', rows)
            matched += len(rows)
        con.commit()
        return {'rows_seen': total_rows, 'rows_indexed': matched, 'output': str(out_db)}
    finally:
        con.close()


def _lookup_times(con: sqlite3.Connection, pairs: list[tuple[int, str]]) -> dict[tuple[int, str], str | None]:
    if not pairs:
        return {}
    unique = list(dict.fromkeys(pairs))
    q = ','.join(['(?,?)'] * len(unique))
    params = [v for pair in unique for v in pair]
    cur = con.execute(f'SELECT cik, accn, acceptance, filingDate FROM acceptance WHERE (cik, accn) IN ({q})', params)
    out: dict[tuple[int, str], str | None] = {}
    for cik, accn, acceptance, filing_date in cur.fetchall():
        out[(int(cik), str(accn))] = acceptance or (f'{filing_date}T00:00:00Z' if filing_date else None)
    return out


def build_pit_fact_events(companyfacts_csv: str | Path, acceptance_db: str | Path, out_csv: str | Path, *, ciks: set[int] | None = None, tags: dict[str, list[str]] | None = None, chunksize: int = 250_000) -> dict:
    """Stream selected Company Facts rows and attach point-in-time information timestamps."""
    inverse = metric_map(tags)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(acceptance_db)
    rows_written = 0
    facts_seen = 0
    facts_selected = 0
    columns = ['cik','entity_name','taxonomy','tag','label','description','unit','start','end','filed','form','frame','fy','fp','accn','val']
    try:
        first = True
        for chunk in pd.read_csv(companyfacts_csv, usecols=columns, chunksize=chunksize, low_memory=False, dtype={'cik':'Int64','accn':'string','tag':'string'}):
            facts_seen += len(chunk)
            chunk = chunk[chunk['tag'].isin(inverse.keys())].copy()
            if ciks is not None:
                chunk = chunk[chunk['cik'].isin(list(ciks))]
            if chunk.empty:
                continue
            facts_selected += len(chunk)
            chunk['metric'] = chunk['tag'].map(inverse)
            chunk['information_time'] = pd.NaT
            pairs = [(int(cik), str(accn)) for cik, accn in chunk[['cik','accn']].dropna().itertuples(index=False, name=None)]
            lookup = _lookup_times(con, pairs)
            times = []
            for cik, accn in zip(chunk['cik'], chunk['accn']):
                key = (int(cik), str(accn)) if pd.notna(cik) and pd.notna(accn) else None
                times.append(lookup.get(key) if key else None)
            chunk['information_time'] = pd.to_datetime(times, utc=True, errors='coerce')
            missing = chunk['information_time'].isna()
            filed = pd.to_datetime(chunk.loc[missing, 'filed'], utc=True, errors='coerce')
            chunk.loc[missing, 'information_time'] = filed
            keep = FACT_FIELDS + ['metric']
            chunk = chunk[keep]
            chunk.to_csv(out_csv, mode='w' if first else 'a', header=first, index=False)
            first = False
            rows_written += len(chunk)
        return {'facts_seen': facts_seen, 'selected_rows': facts_selected, 'rows_written': rows_written, 'output': str(out_csv)}
    finally:
        con.close()


def load_sec_map(path: str | Path) -> set[int]:
    df = pd.read_csv(path, usecols=['cik'], dtype={'cik':'Int64'})
    return set(int(x) for x in df['cik'].dropna().unique())
