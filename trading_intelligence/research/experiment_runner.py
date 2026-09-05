from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from trading_intelligence.research.dataset import build_supervised_market_dataset
from trading_intelligence.research.experiments import run_experiment_suite, save_experiment_report


@dataclass(frozen=True)
class SuiteConfig:
    train_fraction: float = 0.7
    cost_bps_per_side: float = 5.0
    holding_bars: int = 3
    seed: int = 42


def default_experiment_specs(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return feature sets using only columns actually present in the dataset."""
    quant = [c for c in [
        "ret_1", "ret_3", "ret_5", "close_to_ma10",
        "vol_10", "range_pct", "volume_ratio_20",
    ] if c in frame.columns]
    sentiment = [c for c in [
        "sentiment_mean_24h", "sentiment_shock_24h", "news_count_24h",
        "sentiment", "sentiment_change", "event_surprise",
        "novelty", "relevance", "credibility",
    ] if c in frame.columns]
    regime = [c for c in ["volatility_20", "trend_regime", "volatility_regime"] if c in frame.columns]
    specs: dict[str, list[str]] = {}
    if quant:
        specs["quant_only"] = quant
    if sentiment:
        specs["sentiment_only"] = sentiment
    if quant and sentiment:
        specs["quant_plus_sentiment"] = quant + sentiment
    if quant or sentiment:
        combined = list(dict.fromkeys(quant + sentiment + regime))
        specs["combined_regime"] = combined
    if not specs:
        raise ValueError("No recognized feature columns found")
    return specs


def run_suite_from_csv(
    input_csv: str | Path,
    output_json: str | Path,
    config: SuiteConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SuiteConfig()
    frame = pd.read_csv(input_csv)
    if "target_return" not in frame.columns:
        frame = build_supervised_market_dataset(frame, horizon=cfg.holding_bars)
        frame = frame.rename(columns={"forward_return": "target_return"})
    specs = default_experiment_specs(frame)
    results = run_experiment_suite(
        frame,
        specs,
        train_fraction=cfg.train_fraction,
        cost_bps_per_side=cfg.cost_bps_per_side,
        holding_bars=cfg.holding_bars,
        seed=cfg.seed,
    )
    save_experiment_report(results, output_json)
    return results
