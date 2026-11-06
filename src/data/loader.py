"""Data loading and ingestion utilities."""

import os
import logging
from typing import Tuple
import pandas as pd
from sklearn.datasets import fetch_openml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(data_path: str = "data/raw/ames_housing.csv") -> pd.DataFrame:
    """Load the Ames Housing dataset. If not found locally, fetch from OpenML and cache.

    Args:
        data_path: Target path to store or read the raw CSV.

    Returns:
        pd.DataFrame containing the dataset.
    """
    if os.path.exists(data_path):
        logger.info(f"Loading local dataset from: {data_path}")
        return pd.read_csv(data_path)

    logger.info("Local dataset not found. Fetching Ames Housing from OpenML (data_id=42165)...")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Fetch Ames Housing dataset (OpenML ID: 42165 or name 'house_prices')
    dataset = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    df = dataset.frame.copy()

    # Standardize column naming if necessary
    if "SalePrice" not in df.columns and "target" in df.columns:
        df.rename(columns={"target": "SalePrice"}, inplace=True)

    df.to_csv(data_path, index=False)
    logger.info(f"Dataset successfully fetched and saved to {data_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Shape: {df.shape}")
    print(f"Columns preview: {list(df.columns[:10])}")
    print(f"Target head:\n{df['SalePrice'].head()}")
