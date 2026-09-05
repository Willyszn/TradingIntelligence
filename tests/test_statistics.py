import pandas as pd

from trading_intelligence.research.statistics import (
    bootstrap_mean_ci,
    information_coefficient,
    directional_accuracy,
    multiple_testing_warning,
    diagnose_predictions,
)


def test_statistics_are_reasonable():
    actual = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    pred = pd.Series([0.02, -0.01, 0.04, 0.02, -0.02])
    lo, hi = bootstrap_mean_ci(actual, n_boot=200, seed=7)
    assert lo <= actual.mean() <= hi
    assert information_coefficient(pred, actual) > 0.8
    assert directional_accuracy(pred, actual) == 1.0
    diag = diagnose_predictions(pred, actual)
    assert diag.n_observations == 5


def test_multiple_testing_warning():
    assert multiple_testing_warning(9) is None
    assert multiple_testing_warning(12) is not None
