"""Data splitting module with strict leakage prevention and training outlier removal."""

import logging
import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_train_test_splits(
    data_path: str = "data/raw/ames_housing.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.20,
    random_state: int = 42,
    remove_train_outliers: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw data into train and test sets.

    CRITICAL LEAKAGE RULE:
    Outliers are only identified and pruned from the training set *after* the split,
    ensuring the holdout test set represents raw, unbiased operational data.

    Args:
        data_path: Path to raw dataset CSV.
        output_dir: Target directory to save train.parquet and test.parquet.
        test_size: Fraction of samples reserved for test evaluation.
        random_state: Reproducibility seed.
        remove_train_outliers: Whether to prune known high-leverage outliers (>4000 sqft, <$300k).

    Returns:
        Tuple of (train_df, test_df).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # 1. Perform isolated train/test split FIRST
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.info(f"Initial split: Train={train_df.shape}, Test={test_df.shape}")

    # 2. Filter severe high-leverage anomalies from TRAINING SET ONLY
    if remove_train_outliers:
        outlier_condition = (train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000)
        n_outliers = int(outlier_condition.sum())
        if n_outliers > 0:
            logger.info(f"Pruning {n_outliers} high-leverage anomaly properties from TRAINING set.")
            train_df = train_df[~outlier_condition].copy()

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save to parquet if pyarrow is available, otherwise CSV
    try:
        train_path = os.path.join(output_dir, "train.parquet")
        test_path = os.path.join(output_dir, "test.parquet")
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        logger.info(f"Saved processed partitions to {train_path} and {test_path}")
    except (ImportError, Exception):
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved processed partitions to {train_path} and {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    train, test = create_train_test_splits()
    print(f"Final Train shape: {train.shape}, Test shape: {test.shape}")
