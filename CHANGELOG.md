## v4.7 — GDELT event pilot processing

Adds chunked processing of downloaded GDELT event archives into compact daily research partitions, with a SQLite GlobalEventID index for cross-file duplicate detection and a streaming audit. Raw 61-field archives remain untouched as the source of truth.

Build/process with:
```powershell
python scripts\process_gdelt_events.py
python scripts\audit_gdelt_processed.py "$env:TI_DATA_ROOT\news\gdelt_events\processed"
```

## v4.6.0

- Added full 61-field GDELT 2.0 event-schema ingestion.
- Added resumable daily archive downloader, manifest hashing, normalized event/availability timestamps, and GDELT event audit CLI.
- Preserved GDELT availability time for point-in-time research rather than using the inferred event date as information availability.

## v4.3
- Fixed FRED downloader compatibility by omitting open-ended `9999-12-31` observation/realtime parameters; uses the documented open-ended defaults instead.
- Preserved point-in-time real-time history.

## v3.8

- Improved PIT SEC audit duplicate diagnostics: separates economic duplicates from exact filing-event duplicates.
- Economic duplicates are treated as a feature-aggregation precedence issue; exact duplicates are treated as an import/source integrity issue.

# Changelog

## 3.2.0
- Added SEC entity-master extraction from submissions bulk data (CIK, tickers, exchanges, SIC and metadata).
- Added CIK-filtered Company Facts ingestion to avoid expanding the entire SEC bulk archive unnecessarily.
- Added research-universe to SEC CIK mapping utility.
- Added selected-CIK Company Facts importer.
- Preserved accession/filed/acceptance metadata for later point-in-time joins.
- Updated tests; 58/58 pass.
# Changelog

## 3.0.0
- Added configurable liquidity-screened research-universe builder.
- Uses minimum history, price, and median dollar-volume gates with optional cap.
- Produces candidate and selected universe reports without deleting any raw data.
- Preserves all instruments for future research; the screen only defines the initial research subset.


## v3.1.0
- Added research-universe profiles.
- Defaulted the universe CLI to the conservative core_equity_etf profile.
- Added explicit exclusion reasons for known leveraged/inverse symbols.
- Clarified that heuristic classification is not a security master.

## v3.3
- Added explicit `import_companyfacts_for_universe.py` CLI for selective Company Facts import using the SEC universe CIK map.

## 3.4.0
- Added streaming SEC Company Facts audit CLI.
- Keeps raw SEC archives untouched and avoids loading the 9M-row normalized file into memory at once.

## v3.5.0
- Added point-in-time SEC fundamental utilities using accession acceptance timestamps when available.
- Added `build_pit_fundamentals.py` to generate as-of disclosed-fact and snapshot tables.
- Preserves raw Company Facts; research views are derived artifacts.

## 3.7.0
- Added streaming point-in-time SEC event audit.
- Added duplicate-event, timestamp, null, and coverage checks.
- Preserved persistent raw/normalized data architecture.

## 3.9
- Added PIT SEC duplicate diagnostics that distinguish source-exact duplicates from metric-alias collisions and economic duplicates.
- Added `scripts/diagnose_pit_duplicates.py` with bounded duplicate samples.

## v4.0

Added a canonical PIT fundamental observation layer. The new resolver preserves every disclosure revision by keeping `information_time` in the observation key, resolves same-timestamp SEC metric aliases using explicit raw-tag precedence, and keeps debt current/noncurrent/total concepts separate instead of collapsing them prematurely. This layer is intentionally lineage-preserving and is the input to later as-of feature construction.

New script: `scripts/build_pit_fundamental_observations.py`

## v4.1
- Added explicit point-in-time fundamental as-of alignment against caller-supplied UTC market decision timestamps.
- Enforces `information_time <= decision_time`; revisions remain available without collapsing historical disclosures.
- Keeps economic periods separate. TTM/fiscal-period feature construction is deliberately deferred until after this causal alignment layer.
- Added 3 PIT as-of alignment tests.


## v4.3
- Added point-in-time FRED macro vintage acquisition for the initial nine-series macro set.
- Preserves FRED realtime_start/realtime_end revision windows for causal backtesting.
- Adds conservative next-calendar-day availability field to avoid same-day release-time leakage.
- Added macro dataset audit and tests.

## v4.5.0
- Added project-local `.env` loading via `python-dotenv`; process environment variables still take precedence.
- Added `.env.example` and `.gitignore` protection for secrets.
- Added one-command research-data bootstrap script for FRED/ALFRED and optional bounded GDELT event ranges.
- Added PowerShell environment setup and bootstrap helpers.
- Kept GDELT date range explicit to prevent accidental multi-year bulk downloads.
