## v4.7 — GDELT event pilot processing

Adds chunked processing of downloaded GDELT event archives into compact daily research partitions, with a SQLite GlobalEventID index for cross-file duplicate detection and a streaming audit. Raw 61-field archives remain untouched as the source of truth.

Build/process with:
```powershell
python scripts\process_gdelt_events.py
python scripts\audit_gdelt_processed.py "$env:TI_DATA_ROOT\news\gdelt_events\processed"
```

## v4.6 GDELT ingestion

GDELT 2.0 daily event archives are downloaded resumably and retained raw under the persistent data root. The normalized event table keeps both `event_date` (inferred real-world event date) and `availability_time` derived from `DATEADDED`; point-in-time research should join on availability time.

Example:
`python scripts\download_gdelt_events.py --start 2026-01-01 --end 2026-03-31`

Audit extracted event data with `python scripts\audit_gdelt_events.py <csv>`.

# Trading Intelligence v3.1

Stooq bulk importer updated for the real `d_us_txt.zip` format (nested `.us.txt` files with `<TICKER>,<PER>,<DATE>,...` headers). The importer is streaming and writes normalized data to the persistent research root.

Multi-market quantitative + sentiment trading decision-support engine. Human-controlled execution only; live order placement is disabled.

## Persistent data root
Set `TI_DATA_ROOT` to keep datasets independent of versioned project folders. Default: `~/TradingIntelligenceData`.

## Current research milestone
The first empirical pass can use the nine Binance daily crypto datasets in the persistent `market` directory. Sentiment experiments remain gated until point-in-time news/sentiment data is available.

### Useful commands
```powershell
$env:TI_DATA_ROOT="$env:USERPROFILE\TradingIntelligenceData"
python scripts\catalog_market_data.py
python scripts\audit_market_directory.py
python scripts\run_crypto_research.py
```


## Existing-data import (v2.6)
Keep raw research data outside the project at `TI_DATA_ROOT` (default `~/TradingIntelligenceData`).

Stooq archive import:
```powershell
python scripts\import_stooq_archive.py C:\path\to\stooq_us_daily.zip
```

SEC bulk import:
```powershell
python scripts\import_sec_bulk.py --companyfacts C:\path\to\companyfacts.zip --submissions C:\path\to\submissions.zip
```


## Local Stooq + SEC import
The importer streams the large SEC archives instead of loading the entire dataset into RAM.

```powershell
$env:TI_DATA_ROOT="$env:USERPROFILE\TradingIntelligenceData"
python scripts\import_local_datasets.py --stooq C:\Users\willi\Downloads\d_us_txt.zip --companyfacts C:\Users\willi\Downloads\companyfacts.zip --submissions C:\Users\willi\Downloads\submissions.zip
```

Raw source archives remain untouched; normalized outputs are written beneath `TI_DATA_ROOT`.

## v3.0 research universe
The project can scan the persistent market store and build a configurable liquid candidate universe without deleting or rewriting raw data:

```powershell
python scripts\build_research_universe.py
```

Default research screen: at least 750 daily observations, median closing price >= $5, median 90-day dollar volume >= $20M, exclusion of obvious warrant/unit-like ticker patterns, and a top-500 liquidity cap. These are defaults for the *initial research universe*, not trading recommendations; all raw instruments remain available.


## v3.1 research profiles
The universe selector now supports a conservative `core_equity_etf` profile that excludes a maintained set of obvious leveraged/inverse ETF symbols in addition to non-common ticker patterns. This is a research hygiene layer, not an authoritative security-type classification; SEC/security-master metadata will remain the source of truth when imported. The `broad` profile retains these candidates.

## SEC ingestion workflow (v3.2)
1. Import submissions to create `sec_submissions_filings.csv` and `sec_entity_master.csv`.
2. Map the selected Stooq research universe to SEC CIKs with `build_sec_universe_map.py`.
3. Import Company Facts only for mapped CIKs with `import_sec_companyfacts_selected.py`.
This avoids creating a massive Company Facts table for unrelated filers and preserves a clean path to point-in-time filing joins.

### SEC Company Facts selective import
Use `scripts/import_companyfacts_for_universe.py` with the SEC universe mapping to process only mapped CIKs.


## v3.6

Adds a streaming point-in-time SEC event pipeline using a compact accession/acceptance index and selected XBRL tags, avoiding full in-memory joins across the multi-gigabyte SEC archives.

## v3.8
Adds a streaming audit for the selected point-in-time SEC fact-event dataset. Use:
```powershell
$env:TI_DATA_ROOT="$env:USERPROFILE\TradingIntelligenceData"
python scripts\audit_pit_sec.py "$env:TI_DATA_ROOT\processed\pit_sec\selected_pit_fact_events.csv"
```
The audit checks required fields, timestamp parsing, duplicate natural event keys, coverage, and basic filed-vs-information-time consistency without loading the full CSV into memory.


### PIT SEC audit note (v3.8)
The audit reports both economic duplicate keys (which can arise from legitimate repeated/restated facts) and exact duplicate filing-event keys (which are more suspicious). This distinction should be resolved before building aggregated PIT fundamental features.

## v3.9

PIT SEC duplicate diagnostics were expanded. The new `diagnose_pit_duplicates.py` separates true source-row duplicates (including raw SEC tag), metric-alias collisions caused by mapping multiple SEC tags into one research metric, and broader economic duplicates that require precedence during feature aggregation.

## v4.0 — canonical PIT fundamental observations

The v4.0 layer converts the streaming PIT SEC fact-event file into canonical research observations without destroying filing history. Revisions at different `information_time` values remain separate; same-timestamp metric aliases are resolved deterministically using explicit raw SEC tag precedence; and debt current/noncurrent/total concepts remain distinct for later feature engineering.

Build it with:

```powershell
python scripts\build_pit_fundamental_observations.py "$env:TI_DATA_ROOT\processed\pit_sec\selected_pit_fact_events.csv"
```

Default output:
`$env:TI_DATA_ROOT\processed\pit_fundamentals\canonical_observations.csv`

Do not use this file as a daily trading feature panel yet. It is the lineage-preserving canonical observation layer that will feed the as-of/date-aligned feature builder.

### v4.1 PIT fundamental as-of alignment
The new `scripts/build_pit_fundamental_asof.py` aligns canonical SEC observations to explicit decision timestamps. Provide a CSV with `cik,decision_time` where `decision_time` is UTC. The causal rule is strict: only observations with `information_time <= decision_time` are eligible. This layer intentionally does not invent TTM/fiscal-period semantics; those are handled downstream.


## v4.3 — point-in-time macro acquisition
Adds a FRED/ALFRED-compatible macro vintage downloader using the FRED series/observations real-time period. It preserves observation date plus `realtime_start`/`realtime_end` revision windows rather than flattening to today's revised history. The initial set is EFFR, 2Y Treasury, 10Y Treasury, CPI, unemployment, real GDP, high-yield spread, VIX, and broad dollar index.

FRED requires an API key. Keep it in the environment only:

```powershell
$env:FRED_API_KEY="<your key>"
$env:TI_DATA_ROOT="$env:USERPROFILE\TradingIntelligenceData"
python scripts\download_fred_macro.py
python scripts\audit_fred_macro.py "$env:TI_DATA_ROOT\macro\fred_macro_vintages.csv"
```

For causal research, `available_date_conservative` is one calendar day after the FRED real-time start date. This intentionally conservative convention avoids assuming same-day availability until release-time metadata is incorporated.

## Automated local setup (v4.5)

Copy `.env.example` to `.env` and set `FRED_API_KEY` in that file. `.env` is ignored by git and should never be committed. Then run `scripts\\run_bootstrap.ps1`.

To include a bounded GDELT event range in the same run:

```powershell
.\\scripts\\run_bootstrap.ps1 -GdeltStart 2020-01-01 -GdeltEnd 2020-12-31
```

A GDELT range is intentionally explicit because historical daily event archives can become very large. FRED/ALFRED vintage fields are retained for point-in-time research; the FRED observations endpoint supports real-time periods and requires an API key. 
