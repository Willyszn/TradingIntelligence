from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class QuantBaselineModel:
    """Small, reproducible baseline. It predicts forward returns, not trade actions."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=250,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.feature_columns: list[str] = []
        self._fitted = False

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], target_column: str = "target_return") -> None:
        data = frame.dropna(subset=[target_column]).copy()
        if len(data) < 50:
            raise ValueError("Not enough rows to fit the baseline model")
        self.feature_columns = list(feature_columns)
        self.model.fit(data[self.feature_columns], data[target_column])
        self._fitted = True

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        return self.model.predict(frame[self.feature_columns])
