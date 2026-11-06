"""Unit tests for MLOps monitoring and drift detection."""

import numpy as np
import pandas as pd
import pytest
from src.utils.monitoring import calculate_psi, evaluate_feature_drift


def test_calculate_psi_identical_distributions():
    np.random.seed(42)
    dist1 = np.random.normal(0, 1, 1000)
    dist2 = np.random.normal(0, 1, 1000)
    psi = calculate_psi(dist1, dist2)
    assert psi < 0.10, f"Expected minimal drift, got PSI={psi}"


def test_calculate_psi_drifted_distribution():
    np.random.seed(42)
    base = np.random.normal(0, 1, 1000)
    shifted = np.random.normal(3, 1, 1000)  # Significant shift
    psi = calculate_psi(base, shifted)
    assert psi >= 0.25, f"Expected high drift alert, got PSI={psi}"


def test_evaluate_feature_drift():
    df_base = pd.DataFrame({"OverallQual": [5, 6, 7, 8, 9] * 50})
    df_prod = pd.DataFrame({"OverallQual": [5, 6, 7, 8, 9] * 50})
    res = evaluate_feature_drift(df_base, df_prod, ["OverallQual"])
    assert len(res) == 1
    assert res[0]["status"] == "HEALTHY"
