"""Evaluation metrics for regression models."""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_adjusted_r2(r2: float, n: int, p: int) -> float:
    """Calculate Adjusted R-squared.

    Args:
        r2: Standard R-squared score.
        n: Number of samples in the evaluation set.
        p: Number of predictor features (degrees of freedom).

    Returns:
        Adjusted R-squared score.
    """
    if n - p - 1 <= 0:
        return float("nan")
    return 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p: int = 0,
) -> Dict[str, float]:
    """Compute standard regression performance metrics.

    Args:
        y_true: Ground-truth target values (1D array).
        y_pred: Model predictions (1D array).
        p: Number of explanatory features used by the model (for adjusted R2).

    Returns:
        Dictionary containing RMSE, MAE, MAPE (percentage), R2, and Adjusted R2.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Avoid zero division in MAPE
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = float(
            np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
        )
    else:
        mape = float("nan")

    r2 = float(r2_score(y_true, y_pred))
    adj_r2 = calculate_adjusted_r2(r2, n=n, p=p) if p > 0 else r2

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_pct": round(mape, 2),
        "r2": round(r2, 4),
        "adjusted_r2": round(adj_r2, 4),
    }
