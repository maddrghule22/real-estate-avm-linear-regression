"""Unit tests for baseline models and metrics calculation."""

import numpy as np
import pytest
from src.utils.metrics import evaluate_regression, calculate_adjusted_r2
from src.models.baseline import MeanPredictor, MedianPredictor


def test_adjusted_r2() -> None:
    # When R2 is 0.8, n=100, p=5
    adj = calculate_adjusted_r2(0.8, n=100, p=5)
    assert 0.78 < adj < 0.80

    # Overfitted case where n - p - 1 <= 0
    invalid_adj = calculate_adjusted_r2(0.8, n=5, p=5)
    assert np.isnan(invalid_adj)


def test_evaluate_regression_perfect() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    metrics = evaluate_regression(y_true, y_pred, p=1)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["mape_pct"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["adjusted_r2"] == 1.0


def test_mean_predictor() -> None:
    y_train = np.array([10.0, 20.0, 30.0])
    X_train = np.zeros((3, 1))
    X_test = np.zeros((2, 1))

    model = MeanPredictor().fit(X_train, y_train)
    assert model.mean_value_ == 20.0

    preds = model.predict(X_test)
    assert len(preds) == 2
    assert np.all(preds == 20.0)


def test_median_predictor() -> None:
    # Median of [10, 20, 90] is 20 (robust to 90 outlier)
    y_train = np.array([10.0, 20.0, 90.0])
    X_train = np.zeros((3, 1))
    X_test = np.zeros((2, 1))

    model = MedianPredictor().fit(X_train, y_train)
    assert model.median_value_ == 20.0

    preds = model.predict(X_test)
    assert len(preds) == 2
    assert np.all(preds == 20.0)
