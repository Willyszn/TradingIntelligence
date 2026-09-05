from __future__ import annotations

import pandas as pd


def explain_opportunity(row: pd.Series) -> dict:
    """Produce a human-readable evidence summary from already-computed features."""
    positives: list[str] = []
    negatives: list[str] = []
    checks = [
        ("momentum_20", lambda v: v > 0.02, "positive 20-bar momentum", "weak/negative 20-bar momentum"),
        ("momentum_5", lambda v: v > 0.01, "positive short-term momentum", "weak/negative short-term momentum"),
        ("relative_volume_20", lambda v: v > 1.5, "abnormal volume", None),
        ("trend_strength", lambda v: v > 0, "positive trend regime", "negative trend regime"),
        ("sentiment_shock", lambda v: v > 0.2, "positive sentiment shock", "negative sentiment shock"),
        ("event_surprise", lambda v: v > 0.2, "positive event surprise", "negative event surprise"),
    ]
    for col, pred, pos, neg in checks:
        if col not in row or pd.isna(row[col]):
            continue
        if pred(float(row[col])):
            positives.append(pos)
        elif neg:
            negatives.append(neg)
    return {"positives": positives, "negatives": negatives}
