import pandas as pd
from trading_intelligence.research.model_selection import select_research_candidate


def test_selection_gate_rejects_small_or_weak_result():
    df = pd.DataFrame([
        {"name": "a", "rows": 80, "sharpe": 2.0, "mean_return": 0.01},
        {"name": "b", "rows": 120, "sharpe": 0.4, "mean_return": 0.001},
    ])
    d = select_research_candidate(df)
    assert d.candidate == "b"
    assert d.eligible is True
