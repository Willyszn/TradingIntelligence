from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_intelligence.models.baseline import QuantBaselineModel


@dataclass
class ModelPrediction:
    expected_return: float
    probability_positive: float
    model_name: str
    contributions: dict[str, float] = field(default_factory=dict)


class QuantimentEnsemble:
    """Simple research ensemble. Production weights must be learned out-of-sample."""

    def __init__(self, random_state: int = 42) -> None:
        self.quant = QuantBaselineModel(random_state=random_state)
        self.combined = QuantBaselineModel(random_state=random_state + 1)
        self.quant_columns: list[str] = []
        self.combined_columns: list[str] = []
        self._fitted = False

    def fit(self, frame: pd.DataFrame, quant_columns: list[str], combined_columns: list[str]) -> None:
        self.quant.fit(frame, quant_columns)
        self.combined.fit(frame, combined_columns)
        self.quant_columns = list(quant_columns)
        self.combined_columns = list(combined_columns)
        self._fitted = True

    @staticmethod
    def _probability_positive(pred: np.ndarray, scale: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(pred / max(scale, 1e-6), -30, 30)))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Ensemble is not fitted")
        q = self.quant.predict(frame)
        c = self.combined.predict(frame)
        expected = 0.35 * q + 0.65 * c
        scale = float(np.nanstd(expected)) or 1e-3
        return pd.DataFrame({
            "expected_return_quant": q,
            "expected_return_combined": c,
            "expected_return": expected,
            "probability_positive": self._probability_positive(expected, scale),
        }, index=frame.index)
