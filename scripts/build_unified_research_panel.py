from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from trading_intelligence.research.unified_panel import (
    build_unified_research_panel,
)


DEFAULT_DATA_ROOT = Path.home() / "TradingIntelligenceData"

# Leveraged/inverse ETFs are excluded from the ordinary research universe.
LEVERAGED_OR_INVERSE_SYMBOLS = {
    "TQQQ",
    "SQQQ",
    "SOXL",
    "SPXL",
    "UPRO",
    "TNA",
    "FAS",
    "TECL",
    "LABU",
    "CURE",
    "DRN",
    "UDOW",
    "NAIL",
    "QLD",
    "SSO",
    "DDM",
    "ROM",
    "UWM",
    "AGQ",
}

# These ETF symbols are intentionally allowed to have no SEC CompanyFacts CIK.
# They remain in the panel, but SEC-derived features will be unavailable.
SEC_OPTIONAL_ETFS = {
    "IWM.US",
    "SMH.US",
    "SOXX.US",
    "VOO.US",
    "SPY.US",
    "QQQ.US",
    "DIA.US",
    "XLK.US",
    "XLF.US",
    "XLE.US",
    "XLI.US",
    "XLP.US",
    "XLV.US",
    "XLY.US",
    "XLU.US",
}


def _parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def load_selected_universe(
    candidates_csv: str | Path,
    *,
    max_symbols: int | None = 25,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_csv)

    required = {
        "symbol",
        "path",
        "selected",
        "eligible",
    }

    missing = sorted(required - set(candidates.columns))

    if missing:
        raise ValueError(
            f"Research-universe candidates missing columns: {missing}"
        )

    selected = candidates[
        candidates["selected"].astype(str).str.lower().eq("true")
        & candidates["eligible"].astype(str).str.lower().eq("true")
    ].copy()

    selected["symbol"] = selected["symbol"].astype(str).str.upper().str.strip()

    # Remove leveraged/inverse ETFs before applying the symbol cap.
    selected = selected[
        ~selected["symbol"].str.replace(".US", "", regex=False).isin(
            LEVERAGED_OR_INVERSE_SYMBOLS
        )
    ].copy()

    selected = selected.sort_values(
        ["median_dollar_volume_lookback", "rows"],
        ascending=[False, False],
        kind="mergesort",
    )

    if max_symbols is not None:
        selected = selected.head(max_symbols)

    if selected.empty:
        raise ValueError(
            "No selected and eligible research-universe symbols found."
        )

    if selected["symbol"].duplicated().any():
        raise ValueError(
            "Selected research universe contains duplicate symbols."
        )

    return selected.reset_index(drop=True)


def load_market_for_universe(
    universe: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_paths: list[str] = []

    for row in universe.itertuples(index=False):
        path = Path(row.path)

        if not path.exists():
            missing_paths.append(str(path))
            continue

        frame = pd.read_csv(
            path,
            usecols=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
            ],
        )

        frame["symbol"] = str(row.symbol).upper().strip()

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        )

        frame = frame.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ).copy()

        if start is not None:
            frame = frame[frame["timestamp"] >= start]

        if end is not None:
            frame = frame[frame["timestamp"] <= end]

        frames.append(frame)

    if missing_paths:
        raise FileNotFoundError(
            "Missing market files:\n"
            + "\n".join(missing_paths[:20])
        )

    if not frames:
        raise ValueError(
            "No market rows were loaded for the selected universe."
        )

    market = pd.concat(frames, ignore_index=True)

    market = market.drop_duplicates(
        ["symbol", "timestamp"],
        keep="last",
    )

    market = market.sort_values(
        ["symbol", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    if market.empty:
        raise ValueError(
            "Market data is empty after date filtering."
        )

    return market


def load_security_master(
    sec_map_csv: str | Path,
    symbols: Iterable[str],
) -> pd.DataFrame:
    mapping = pd.read_csv(sec_map_csv)

    required = {
        "symbol",
        "cik",
        "mapping_status",
    }

    missing = sorted(required - set(mapping.columns))

    if missing:
        raise ValueError(
            f"SEC universe map missing columns: {missing}"
        )

    mapping["symbol"] = (
        mapping["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    symbols_set = {
        str(symbol).upper().strip()
        for symbol in symbols
    }

    master = mapping[
        mapping["symbol"].isin(symbols_set)
    ].copy()

    master["cik"] = pd.to_numeric(
        master["cik"],
        errors="coerce",
    ).astype("Int64")

    # Validate any SEC mapping that does exist.
    bad = master[
        ~master["mapping_status"].astype(str).eq("unique")
        | (
            master["cik"].isna()
            & ~master["symbol"].isin(SEC_OPTIONAL_ETFS)
        )
    ]

    if not bad.empty:
        bad_symbols = sorted(
            bad["symbol"].astype(str).unique()
        )

        raise ValueError(
            "Selected universe has non-unique or missing SEC mappings "
            f"for non-optional securities: {bad_symbols[:20]}"
        )

    master = (
        master[
            ["symbol", "cik"]
        ]
        .drop_duplicates("symbol")
    )

    # Add explicitly recognised ETFs which have no SEC CIK mapping.
    missing_symbols = sorted(
        symbols_set - set(master["symbol"])
    )

    unknown_missing = [
        symbol
        for symbol in missing_symbols
        if symbol not in SEC_OPTIONAL_ETFS
    ]

    if unknown_missing:
        raise ValueError(
            "Selected universe is missing SEC mappings for "
            f"unrecognised securities: {unknown_missing[:20]}"
        )

    if missing_symbols:
        optional_rows = pd.DataFrame(
            {
                "symbol": missing_symbols,
                "cik": pd.Series(
                    [pd.NA] * len(missing_symbols),
                    dtype="Int64",
                ),
            }
        )

        master = pd.concat(
            [master, optional_rows],
            ignore_index=True,
        )

    master = (
        master
        .drop_duplicates("symbol")
        .sort_values("symbol", kind="mergesort")
        .reset_index(drop=True)
    )

    return master


def load_sec_for_ciks(
    sec_csv: str | Path,
    ciks: set[int],
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    if not ciks:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        sec_csv,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["cik"] = pd.to_numeric(
            chunk["cik"],
            errors="coerce",
        ).astype("Int64")

        filtered = chunk[
            chunk["cik"].isin(ciks)
        ].copy()

        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def load_fred(
    fred_csv: str | Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        fred_csv,
        low_memory=False,
    )

    # The persisted FRED dataset uses observation_date.
    # The downstream PIT alignment currently expects date.
    if "date" not in frame.columns:
        if "observation_date" in frame.columns:
            frame = frame.rename(
                columns={"observation_date": "date"}
            )
        else:
            raise ValueError(
                "FRED data requires either 'date' or "
                "'observation_date'."
            )

    required = {
        "series_id",
        "date",
        "value",
        "realtime_start",
        "realtime_end",
        "available_date_conservative",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"FRED data missing columns: {missing}"
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        utc=True,
        errors="coerce",
    )

    frame["realtime_start"] = pd.to_datetime(
        frame["realtime_start"],
        utc=True,
        errors="coerce",
    )

    frame["realtime_end"] = pd.to_datetime(
        frame["realtime_end"],
        utc=True,
        errors="coerce",
    )

    frame["available_date_conservative"] = pd.to_datetime(
        frame["available_date_conservative"],
        utc=True,
        errors="coerce",
    )

    return frame


def build_report(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    security_master: pd.DataFrame,
    sec_rows: int,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> dict[str, object]:
    feature_columns = [
        column
        for column in panel.columns
        if column.startswith("sec_")
        or column.startswith("macro_")
    ]

    feature_missing = {
        column: float(
            panel[column].isna().mean()
        )
        for column in feature_columns
    }

    return {
        "symbols_requested": int(len(universe)),
        "symbols_in_panel": int(
            panel["symbol"].nunique()
        ),
        "panel_rows": int(len(panel)),
        "panel_columns": int(
            len(panel.columns)
        ),
        "market_first_timestamp": (
            panel["timestamp"].min().isoformat()
        ),
        "market_last_timestamp": (
            panel["timestamp"].max().isoformat()
        ),
        "decision_first_timestamp": (
            panel["decision_time"].min().isoformat()
        ),
        "decision_last_timestamp": (
            panel["decision_time"].max().isoformat()
        ),
        "entry_first_timestamp": (
            panel["entry_timestamp"].min().isoformat()
        ),
        "target_last_timestamp": (
            panel["target_timestamp"].max().isoformat()
        ),
        "sec_mapped_symbols": int(
            security_master["cik"].notna().sum()
        ),
        "sec_ciks": int(
            security_master["cik"].dropna().nunique()
        ),
        "sec_rows_loaded": int(sec_rows),
        "fred_feature_count": int(
            sum(
                column.startswith("macro_")
                and not column.endswith("_age_days")
                for column in panel.columns
            )
        ),
        "sec_feature_count": int(
            sum(
                column.startswith("sec_")
                for column in panel.columns
            )
        ),
        "label_available_rate": float(
            panel["target_return"].notna().mean()
        ),
        "start_filter": (
            None
            if start is None
            else start.isoformat()
        ),
        "end_filter": (
            None
            if end is None
            else end.isoformat()
        ),
        "feature_missing_fraction": feature_missing,
    }


def build_panel(
    *,
    data_root: str | Path,
    candidates_csv: str | Path | None = None,
    sec_map_csv: str | Path | None = None,
    sec_csv: str | Path | None = None,
    fred_csv: str | Path | None = None,
    max_symbols: int | None = 25,
    start: str | None = None,
    end: str | None = None,
    chunksize: int = 250_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    data_root = Path(data_root)
    reports_root = data_root / "reports"

    candidates_csv = Path(
        candidates_csv
        or reports_root / "research_universe_candidates.csv"
    )

    sec_map_csv = Path(
        sec_map_csv
        or reports_root / "sec_universe_map.csv"
    )

    sec_csv = Path(
        sec_csv
        or data_root
        / "processed"
        / "pit_fundamentals"
        / "canonical_observations.csv"
    )

    fred_csv = Path(
        fred_csv
        or data_root
        / "macro"
        / "fred_macro_vintages.csv"
    )

    start_ts = _parse_date(start)
    end_ts = _parse_date(end)

    universe = load_selected_universe(
        candidates_csv,
        max_symbols=max_symbols,
    )

    market = load_market_for_universe(
        universe,
        start=start_ts,
        end=end_ts,
    )

    master = load_security_master(
        sec_map_csv,
        market["symbol"].unique(),
    )

    ciks = set(
        master.loc[
            master["cik"].notna(),
            "cik",
        ]
        .astype(int)
    )

    sec = load_sec_for_ciks(
        sec_csv,
        ciks,
        chunksize=chunksize,
    )

    # SEC is optional for ETF rows, but common-equity rows still
    # need actual SEC observations.
    common_symbols_missing_sec = sorted(
        set(
            master.loc[
                master["cik"].isna(),
                "symbol",
            ]
        )
        - SEC_OPTIONAL_ETFS
    )

    if common_symbols_missing_sec:
        raise ValueError(
            "Common-equity securities have no SEC mapping: "
            f"{common_symbols_missing_sec[:20]}"
        )

    fred = load_fred(fred_csv)

    panel = build_unified_research_panel(
        market,
        sec_observations=sec,
        fred_observations=fred,
        security_master=master,
        market_session="us_equity_daily",
    )

    report = build_report(
        panel,
        universe,
        master,
        len(sec),
        start=start_ts,
        end=end_ts,
    )

    return panel, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the point-in-time US-equity unified "
            "research panel."
        )
    )

    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
    )

    parser.add_argument(
        "--max-symbols",
        type=int,
        default=25,
        help=(
            "Number of selected universe symbols. "
            "Use 500 for the full selected universe."
        ),
    )

    parser.add_argument("--start")
    parser.add_argument("--end")

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Panel CSV output path. Defaults under "
            "processed/research."
        ),
    )

    parser.add_argument(
        "--report",
        default=None,
        help=(
            "JSON readiness/report output path. "
            "Defaults under processed/research."
        ),
    )

    parser.add_argument(
        "--sec-chunksize",
        type=int,
        default=250_000,
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)

    output_dir = (
        data_root
        / "processed"
        / "research"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "full"
        if args.max_symbols >= 500
        else f"top{args.max_symbols}"
    )

    out_path = (
        Path(args.out)
        if args.out
        else (
            output_dir
            / f"unified_daily_panel_{suffix}.csv"
        )
    )

    report_path = (
        Path(args.report)
        if args.report
        else (
            output_dir
            / f"unified_daily_panel_{suffix}_report.json"
        )
    )

    panel, report = build_panel(
        data_root=data_root,
        max_symbols=args.max_symbols,
        start=args.start,
        end=args.end,
        chunksize=args.sec_chunksize,
    )

    panel.to_csv(
        out_path,
        index=False,
    )

    report.update(
        {
            "panel_output": str(out_path),
            "report_output": str(report_path),
        }
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())