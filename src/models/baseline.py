"""Baseline models for regression benchmark."""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.metrics import evaluate_regression

logger = logging.getLogger(__name__)


class MeanPredictor:
    """Baseline regressor that always predicts the training target mean."""

    def __init__(self) -> None:
        self.mean_value_: float = 0.0

    def fit(self, X: Any, y: pd.Series | np.ndarray) -> "MeanPredictor":
        """Fit the mean predictor by calculating the sample mean."""
        self.mean_value_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict constant mean for all samples."""
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.mean_value_)


class MedianPredictor:
    """Baseline regressor that always predicts the training target median.

    Robust against heavy target skewness and extreme outliers.
    """

    def __init__(self) -> None:
        self.median_value_: float = 0.0

    def fit(self, X: Any, y: pd.Series | np.ndarray) -> "MedianPredictor":
        """Fit the median predictor by calculating the sample median."""
        self.median_value_ = float(np.median(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict constant median for all samples."""
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.median_value_)


def evaluate_baselines(
    df: pd.DataFrame,
    target_col: str = "SalePrice",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Split dataset and calculate baseline performance for mean and median predictors.

    Args:
        df: Input DataFrame containing features and target.
        target_col: Name of the continuous target variable.
        test_size: Fraction of samples reserved for test evaluation.
        random_state: Reproducibility seed.

    Returns:
        Tuple of (mean_metrics, median_metrics) dictionaries.
    """
    y = df[target_col].dropna().values
    X = np.zeros((len(y), 1))  # Dummy placeholder feature matrix

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 1. Mean Predictor
    mean_model = MeanPredictor().fit(X_train, y_train)
    y_pred_mean = mean_model.predict(X_test)
    mean_metrics = evaluate_regression(y_test, y_pred_mean, p=0)

    # 2. Median Predictor
    median_model = MedianPredictor().fit(X_train, y_train)
    y_pred_median = median_model.predict(X_test)
    median_metrics = evaluate_regression(y_test, y_pred_median, p=0)

    return mean_metrics, median_metrics


if __name__ == "__main__":
    from src.data.loader import load_raw_data

    raw_df = load_raw_data()
    mean_res, median_res = evaluate_baselines(raw_df)
    print("Baseline Evaluation Results on Test Set (20%):")
    print(f"Mean Predictor:   {mean_res}")
    print(f"Median Predictor: {median_res}")
