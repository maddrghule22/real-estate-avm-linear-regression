"""Unit tests for assumption checks and diagnostic routines."""

import numpy as np
import pandas as pd
import pytest
from src.utils.diagnostics import compute_vif, run_assumption_diagnostics


def test_compute_vif_perfect_collinearity():
    # Construct a synthetic DataFrame with collinear features
    x1 = np.linspace(1, 100, 100)
    x2 = 2.0 * x1 + np.random.normal(0, 0.001, 100)
    x3 = np.random.normal(0, 1, 100)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    vif_df = compute_vif(df, threshold=10.0, output_path="reports/test_vif.csv")
    assert len(vif_df) == 3
    # x1 and x2 should have high VIF
    vif_dict = dict(zip(vif_df["Feature"], vif_df["VIF"]))
    assert vif_dict["x1"] > 10.0
    assert vif_dict["x2"] > 10.0
    assert vif_dict["x3"] < 5.0


def test_run_assumption_diagnostics():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    # Synthetic clean linear relationship
    y_true = 2.0 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + np.random.normal(0, 0.2, n)
    y_pred = 2.0 + 1.5 * X[:, 0] - 0.8 * X[:, 1]

    diag = run_assumption_diagnostics(y_true, y_pred, X)
    assert "durbin_watson" in diag
    assert "breusch_pagan_homoscedasticity" in diag
    assert "jarque_bera_normality" in diag
    # Synthetic normal independent noise should yield DW near 2.0
    assert 1.7 < diag["durbin_watson"]["statistic"] < 2.3
