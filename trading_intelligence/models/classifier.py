from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class DirectionClassifier:
    """Predicts probability of a positive forward return without producing a trade action."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
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
        y = (data[target_column].astype(float) > 0).astype(int)
        if y.nunique() < 2:
            raise ValueError("Direction classifier requires both positive and non-positive target classes")
        self.feature_columns = list(feature_columns)
        self.model.fit(data[self.feature_columns], y)
        self._fitted = True

    def predict_proba_positive(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier is not fitted")
        return self.model.predict_proba(frame[self.feature_columns])[:, 1]
