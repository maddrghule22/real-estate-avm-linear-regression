"""Integration tests for trained model artifacts and inference."""

import os
import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_data():
    df = pd.read_parquet("data/processed/test.parquet")
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"].values
    return X, y


def test_best_linear_pipeline_load_and_predict() -> None:
    model_path = "models/best_linear_pipeline.joblib"
    assert os.path.exists(model_path), f"Artifact missing: {model_path}"

    pipeline = joblib.load(model_path)
    df_test = pd.read_parquet("data/processed/test.parquet")
    X_test = df_test.drop(columns=["SalePrice"])

    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert np.all(preds > 0), "Home price predictions must strictly be positive numbers"
    assert not np.isnan(preds).any(), "Predictions must not contain NaN values"

    # Sanity check: Mean prediction should fall in reasonable real estate range ($100k - $300k)
    mean_val = float(np.mean(preds))
    assert 140000.0 < mean_val < 240000.0, f"Unrealistic average prediction: {mean_val}"
