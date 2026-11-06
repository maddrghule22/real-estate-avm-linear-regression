"""Feature engineering transformers and pipeline builder."""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Core selected categorical and numerical feature lists based on EDA
CATEGORICAL_FEATURES = [
    "Neighborhood",
    "BldgType",
    "HouseStyle",
    "ExterQual",
    "KitchenQual",
    "Foundation",
    "HeatingQC",
    "CentralAir",
    "SaleCondition",
]

NUMERICAL_BASE_FEATURES = [
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "YrSold",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    "LotArea",
]


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to compute domain-specific real estate features and interactions."""

    def __init__(self, include_interactions: bool = True) -> None:
        self.include_interactions = include_interactions

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "DomainFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer domain features without modifying input inplace."""
        X_out = X.copy()

        # 1. Total Square Footage (Basement + 1st Floor + 2nd Floor)
        bsmt_sf = X_out["TotalBsmtSF"].fillna(0)
        flr1_sf = X_out["1stFlrSF"].fillna(0)
        flr2_sf = X_out["2ndFlrSF"].fillna(0)
        X_out["TotalSF"] = bsmt_sf + flr1_sf + flr2_sf

        # 2. Total Bathrooms (Full = 1.0, Half = 0.5)
        full_baths = X_out["FullBath"].fillna(0)
        half_baths = X_out["HalfBath"].fillna(0) * 0.5
        bsmt_full = X_out["BsmtFullBath"].fillna(0)
        bsmt_half = X_out["BsmtHalfBath"].fillna(0) * 0.5
        X_out["TotalBaths"] = full_baths + half_baths + bsmt_full + bsmt_half

        # 3. Property Age and Remodel Age at Sale
        yr_sold = X_out["YrSold"].fillna(2010)
        yr_built = X_out["YearBuilt"].fillna(yr_sold)
        yr_remod = X_out["YearRemodAdd"].fillna(yr_built)

        X_out["HouseAge"] = (yr_sold - yr_built).clip(lower=0)
        X_out["RemodAge"] = (yr_sold - yr_remod).clip(lower=0)

        # 4. Binary Amenity Flags
        garage_area = X_out["GarageArea"].fillna(0)
        fireplaces = X_out["Fireplaces"].fillna(0)
        X_out["HasGarage"] = (garage_area > 0).astype(float)
        X_out["HasBasement"] = (bsmt_sf > 0).astype(float)
        X_out["HasFireplace"] = (fireplaces > 0).astype(float)

        # 5. Domain Interaction: Overall Quality x Total Square Footage
        # (Marginal value of square footage increases with higher finish quality)
        if self.include_interactions:
            qual = X_out["OverallQual"].fillna(5)
            X_out["Quality_x_TotalSF"] = qual * X_out["TotalSF"]

        return X_out


# Engineered numerical features that will be produced by DomainFeatureEngineer
ALL_NUMERICAL_FEATURES = [
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "LotArea",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageArea",
    "TotalSF",
    "TotalBaths",
    "HouseAge",
    "RemodAge",
    "HasGarage",
    "HasBasement",
    "HasFireplace",
    "Quality_x_TotalSF",
]


def build_preprocessor_pipeline(
    categorical_features: List[str] = CATEGORICAL_FEATURES,
    numerical_features: List[str] = ALL_NUMERICAL_FEATURES,
    drop_first: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer for preprocessing numerical and categorical features.

    Args:
        categorical_features: List of categorical column names.
        numerical_features: List of numerical column names.
        drop_first: If True, drops first dummy variable in OneHotEncoder to avoid the dummy variable trap.

    Returns:
        Configured ColumnTransformer pipeline.
    """
    # Numerical sub-pipeline: Median Imputation -> Standard Scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical sub-pipeline: Constant 'Missing' Imputation -> One-Hot Encoding
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first" if drop_first else None,
                    handle_unknown="ignore" if not drop_first else "infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def build_full_feature_pipeline(drop_first: bool = True) -> Pipeline:
    """Construct complete end-to-end feature transformation pipeline."""
    return Pipeline(
        steps=[
            ("domain_features", DomainFeatureEngineer(include_interactions=True)),
            ("preprocessor", build_preprocessor_pipeline(drop_first=drop_first)),
        ]
    )
