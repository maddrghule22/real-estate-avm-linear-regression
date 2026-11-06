"""Unit tests for feature engineering transformers and pipelines."""

import numpy as np
import pandas as pd
import pytest
from src.features.transformers import (
    DomainFeatureEngineer,
    build_preprocessor_pipeline,
    build_full_feature_pipeline,
)


@pytest.fixture
def sample_property_df() -> pd.DataFrame:
    """Create synthetic property records for pipeline testing."""
    return pd.DataFrame(
        {
            "OverallQual": [7, 5, 8],
            "OverallCond": [5, 6, 5],
            "YearBuilt": [2000, 1970, 2010],
            "YearRemodAdd": [2005, 1970, 2015],
            "YrSold": [2008, 2010, 2020],
            "TotalBsmtSF": [800.0, 600.0, 1200.0],
            "1stFlrSF": [900.0, 700.0, 1300.0],
            "2ndFlrSF": [500.0, 0.0, 800.0],
            "GrLivArea": [1400.0, 700.0, 2100.0],
            "FullBath": [2, 1, 2],
            "HalfBath": [1, 0, 1],
            "BsmtFullBath": [1, 0, 1],
            "BsmtHalfBath": [0, 0, 0],
            "TotRmsAbvGrd": [7, 4, 9],
            "Fireplaces": [1, 0, 2],
            "GarageCars": [2, 1, 3],
            "GarageArea": [450.0, 250.0, 700.0],
            "LotArea": [8000.0, 5000.0, 12000.0],
            "Neighborhood": ["CollgCr", "OldTown", "NridgHt"],
            "BldgType": ["1Fam", "1Fam", "1Fam"],
            "HouseStyle": ["2Story", "1Story", "2Story"],
            "ExterQual": ["Gd", "TA", "Ex"],
            "KitchenQual": ["Gd", "TA", "Ex"],
            "Foundation": ["PConc", "CBlock", "PConc"],
            "HeatingQC": ["Ex", "TA", "Ex"],
            "CentralAir": ["Y", "N", "Y"],
            "SaleCondition": ["Normal", "Normal", "Partial"],
        }
    )


def test_domain_feature_engineer(sample_property_df: pd.DataFrame) -> None:
    engineer = DomainFeatureEngineer(include_interactions=True)
    transformed = engineer.transform(sample_property_df)

    # 1. TotalSF = TotalBsmtSF (800) + 1stFlrSF (900) + 2ndFlrSF (500) = 2200
    assert transformed.loc[0, "TotalSF"] == 2200.0

    # 2. TotalBaths = Full (2) + 0.5*Half (0.5) + BsmtFull (1) + 0.5*BsmtHalf (0) = 3.5
    assert transformed.loc[0, "TotalBaths"] == 3.5

    # 3. HouseAge = YrSold (2008) - YearBuilt (2000) = 8
    assert transformed.loc[0, "HouseAge"] == 8

    # 4. Binary Flags
    assert transformed.loc[0, "HasGarage"] == 1.0
    assert transformed.loc[0, "HasBasement"] == 1.0
    assert transformed.loc[0, "HasFireplace"] == 1.0

    # 5. Interaction: Quality (7) * TotalSF (2200) = 15400
    assert transformed.loc[0, "Quality_x_TotalSF"] == 7 * 2200.0


def test_full_pipeline_shape_and_no_leakage(sample_property_df: pd.DataFrame) -> None:
    pipeline = build_full_feature_pipeline(drop_first=False)
    transformed = pipeline.fit_transform(sample_property_df)

    # Output should be 2D numpy array without NaNs
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()

    # Out-of-vocabulary handling for unseen categorical levels
    new_unseen = sample_property_df.copy()
    new_unseen.loc[0, "Neighborhood"] = "BrandNewNeighborhoodNeverSeen"
    transformed_unseen = pipeline.transform(new_unseen)
    assert not np.isnan(transformed_unseen).any()
