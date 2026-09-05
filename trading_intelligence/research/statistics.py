from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


def _clean(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_mean_ci(values: pd.Series | np.ndarray, *, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the sample mean."""
    x = _clean(values)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def information_coefficient(predicted: pd.Series, actual: pd.Series) -> float:
    """Pearson correlation between predictions and realized returns."""
    a = pd.to_numeric(predicted, errors="coerce")
    b = pd.to_numeric(actual, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(a[mask].corr(b[mask]))


def directional_accuracy(predicted: pd.Series, actual: pd.Series) -> float:
    a = pd.to_numeric(predicted, errors="coerce")
    b = pd.to_numeric(actual, errors="coerce")
    mask = a.notna() & b.notna() & (b != 0)
    if not mask.any():
        return float("nan")
    return float(((a[mask] > 0) == (b[mask] > 0)).mean())


@dataclass(frozen=True)
class ResearchDiagnostics:
    mean_return: float
    mean_return_ci_low: float
    mean_return_ci_high: float
    information_coefficient: float
    directional_accuracy: float
    n_observations: int

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose_predictions(predicted: pd.Series, actual: pd.Series, *, seed: int = 42) -> ResearchDiagnostics:
    x = pd.to_numeric(actual, errors="coerce")
    ci_low, ci_high = bootstrap_mean_ci(x, seed=seed)
    return ResearchDiagnostics(
        mean_return=float(x.mean()),
        mean_return_ci_low=ci_low,
        mean_return_ci_high=ci_high,
        information_coefficient=information_coefficient(predicted, actual),
        directional_accuracy=directional_accuracy(predicted, actual),
        n_observations=int(x.notna().sum()),
    )


def multiple_testing_warning(n_experiments: int, *, alpha: float = 0.05) -> str | None:
    """Warn when many experiments make naive p-value interpretation unreliable."""
    if n_experiments < 10:
        return None
    return (
        f"{n_experiments} model variants were evaluated. Treat apparent winners cautiously: "
        f"multiple testing raises false-discovery risk. Confirm the selected strategy on a fresh holdout "
        f"before relying on it (nominal alpha={alpha:.2f})."
    )
