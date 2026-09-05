import numpy as np
import pandas as pd

from trading_intelligence.research.experiment_runner import default_experiment_specs
from trading_intelligence.research.experiments import run_experiment_suite


def _frame(n=180):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
        "symbol": ["AAA"] * n,
        "open": 100 + np.cumsum(rng.normal(0, 1, n)),
        "close": 100 + np.cumsum(rng.normal(0, 1, n)),
        "target_return": rng.normal(0, 0.02, n),
        "ret_1": rng.normal(0, 0.01, n),
        "ret_5": rng.normal(0, 0.02, n),
        "close_to_ma10": rng.normal(0, 0.01, n),
        "range_pct": rng.normal(0, 0.02, n),
        "vol_10": rng.uniform(0.01, 0.05, n),
        "volume_ratio_20": rng.uniform(0.5, 2.0, n),
        "volume_z": rng.normal(0, 1, n),
        "sentiment": rng.normal(0, 1, n),
        "sentiment_change": rng.normal(0, 1, n),
        "sentiment_shock": rng.normal(0, 1, n),
        "event_surprise": rng.normal(0, 1, n),
        "novelty": rng.uniform(0, 1, n),
        "relevance": rng.uniform(0, 1, n),
        "credibility": rng.uniform(0, 1, n),
    })


def test_default_specs_include_expected_layers():
    specs = default_experiment_specs(_frame())
    assert "quant_only" in specs
    assert "sentiment_only" in specs
    assert "quant_plus_sentiment" in specs
    assert "combined_regime" in specs


def test_suite_returns_ranked_dataframe():
    frame = _frame()
    specs = default_experiment_specs(frame)
    result = run_experiment_suite(frame, specs, train_fraction=0.7, holding_bars=3)
    assert len(result) == 4
    assert list(result.columns) == [
        "name", "rows", "mean_return", "sharpe", "max_drawdown", "win_rate", "directional_accuracy"
    ]
