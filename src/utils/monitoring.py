"""MLOps drift detection and model monitoring utilities."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between baseline and production feature distributions.

    Rules of Thumb:
      - PSI < 0.10: No significant distribution drift; model is stable.
      - 0.10 <= PSI < 0.25: Moderate drift; flag for review.
      - PSI >= 0.25: Severe drift; trigger automated model retraining.

    Args:
        expected: Baseline reference distribution (e.g., training split).
        actual: Incoming production serving distribution.
        num_buckets: Number of quantile buckets.
        epsilon: Small smoothing constant to prevent division by zero.

    Returns:
        Population Stability Index score.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    # Calculate quantile breakpoints using expected baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    breakpoints = np.percentile(expected, quantiles)
    breakpoints[0] -= epsilon
    breakpoints[-1] += epsilon

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    # Normalize back to sum to 1
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 4))


def evaluate_feature_drift(
    baseline_df: pd.DataFrame,
    production_df: pd.DataFrame,
    features: List[str],
) -> List[Dict[str, object]]:
    """Run KS-test and PSI across numerical features to evaluate data drift."""
    results = []
    for feat in features:
        if feat not in baseline_df.columns or feat not in production_df.columns:
            continue

        base_vals = baseline_df[feat].dropna().values
        prod_vals = production_df[feat].dropna().values

        # Kolmogorov-Smirnov Test
        ks_stat, ks_pvalue = stats.ks_2samp(base_vals, prod_vals)
        psi_score = calculate_psi(base_vals, prod_vals)

        status = "HEALTHY"
        if psi_score >= 0.25 or ks_pvalue < 0.01:
            status = "RETRAIN_TRIGGERED"
        elif psi_score >= 0.10:
            status = "WARNING_DRIFT"

        results.append(
            {
                "feature": feat,
                "psi": psi_score,
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 4),
                "status": status,
            }
        )

    return results
