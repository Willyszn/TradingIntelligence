from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_COLS = [
    'cik','entity_name','taxonomy','tag','unit','start','end','filed','form',
    'frame','fy','fp','accn','val','information_time','metric'
]


def diagnose_pit_duplicates(path: str | Path, *, chunksize: int = 250_000, sample_limit: int = 25) -> dict:
    path = Path(path)
    rows = 0
    chunks = 0
    seen_source: set[tuple] = set()
    seen_metric_alias: set[tuple] = set()
    seen_economic: set[tuple] = set()
    source_dupes = metric_alias_dupes = economic_dupes = 0
    samples: list[dict] = []

    usecols = [
        'cik','entity_name','taxonomy','tag','unit','start','end','filed','form',
        'frame','fy','fp','accn','val','information_time','metric'
    ]
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=lambda c: c in usecols):
        chunks += 1
        rows += len(chunk)
        src_cols = [c for c in [
            'cik','taxonomy','tag','unit','start','end','filed','form',
            'frame','fy','fp','accn','val','information_time'
        ] if c in chunk]
        alias_cols = [c for c in ['cik','metric','unit','start','end','information_time','accn'] if c in chunk]
        econ_cols = [c for c in ['cik','metric','unit','start','end','information_time'] if c in chunk]

        for row in chunk[src_cols].astype('string').itertuples(index=False, name=None):
            key = tuple(row)
            if key in seen_source:
                source_dupes += 1
                if len(samples) < sample_limit:
                    samples.append({'kind': 'source_exact', 'key': dict(zip(src_cols, row))})
            else:
                seen_source.add(key)

        for row in chunk[alias_cols].astype('string').itertuples(index=False, name=None):
            key = tuple(row)
            if key in seen_metric_alias:
                metric_alias_dupes += 1
            else:
                seen_metric_alias.add(key)

        for row in chunk[econ_cols].astype('string').itertuples(index=False, name=None):
            key = tuple(row)
            if key in seen_economic:
                economic_dupes += 1
            else:
                seen_economic.add(key)

    return {
        'rows': rows,
        'chunks': chunks,
        'source_exact_duplicate_rows': source_dupes,
        'metric_alias_duplicate_rows': metric_alias_dupes,
        'economic_duplicate_rows': economic_dupes,
        'source_exact_duplicate_rate': source_dupes / rows if rows else 0.0,
        'metric_alias_duplicate_rate': metric_alias_dupes / rows if rows else 0.0,
        'economic_duplicate_rate': economic_dupes / rows if rows else 0.0,
        'source_exact_samples': samples,
    }
