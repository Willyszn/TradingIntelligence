from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from trading_intelligence.research.dataset import load_market_csv, build_supervised_market_dataset, join_point_in_time_news
from trading_intelligence.research.readiness import DatasetReadiness, assess_market_dataset


@dataclass(frozen=True)
class AssemblyResult:
    raw_rows: int
    processed_rows: int
    symbols: int
    market_files: int
    news_rows: int
    readiness: DatasetReadiness


def discover_market_files(market_dir: str | Path) -> list[Path]:
    """Return deterministic normalized market CSVs from a directory tree."""
    root = Path(market_dir)
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def load_market_directory(market_dir: str | Path) -> pd.DataFrame:
    paths = discover_market_files(market_dir)
    if not paths:
        raise FileNotFoundError(f"No market CSV files found under {market_dir}")
    frames = [load_market_csv(p) for p in paths]
    market = pd.concat(frames, ignore_index=True)
    market = (
        market.drop_duplicates(["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    return market


def assemble_research_dataset(
    market_dir: str | Path,
    *,
    news_csv: str | Path | None = None,
    horizon: int = 3,
    news_lookback_hours: int = 24,
) -> tuple[pd.DataFrame, AssemblyResult]:
    market = load_market_directory(market_dir)
    dataset = build_supervised_market_dataset(market, horizon=horizon)
    news_rows = 0
    if news_csv is not None:
        news = pd.read_csv(news_csv)
        news_rows = len(news)
        dataset = join_point_in_time_news(dataset, news, lookback_hours=news_lookback_hours)
    readiness = assess_market_dataset(market)
    result = AssemblyResult(
        raw_rows=int(len(market)),
        processed_rows=int(len(dataset)),
        symbols=int(market["symbol"].nunique()),
        market_files=len(discover_market_files(market_dir)),
        news_rows=int(news_rows),
        readiness=readiness,
    )
    return dataset, result
