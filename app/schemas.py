"""Pydantic request and response schemas for property valuation microservice."""

from typing import List
from pydantic import BaseModel, ConfigDict, Field


class PropertyInput(BaseModel):
    """Input payload representing residential property attributes."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "YrSold": 2008,
                "TotalBsmtSF": 856.0,
                "1stFlrSF": 856.0,
                "2ndFlrSF": 854.0,
                "GrLivArea": 1710.0,
                "FullBath": 2,
                "HalfBath": 1,
                "BsmtFullBath": 1,
                "BsmtHalfBath": 0,
                "TotRmsAbvGrd": 8,
                "Fireplaces": 1,
                "GarageCars": 2,
                "GarageArea": 548.0,
                "LotArea": 8450.0,
                "Neighborhood": "CollgCr",
                "BldgType": "1Fam",
                "HouseStyle": "2Story",
                "ExterQual": "Gd",
                "KitchenQual": "Gd",
                "Foundation": "PConc",
                "HeatingQC": "Ex",
                "CentralAir": "Y",
                "SaleCondition": "Normal",
            }
        },
    )

    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality rating (1-10)")
    OverallCond: int = Field(5, ge=1, le=10, description="Overall condition rating (1-10)")
    YearBuilt: int = Field(..., ge=1800, le=2030, description="Original construction date")
    YearRemodAdd: int = Field(..., ge=1800, le=2030, description="Remodel date (same as construction date if no remodel)")
    YrSold: int = Field(2010, ge=2000, le=2035, description="Year of sale transaction")
    TotalBsmtSF: float = Field(..., ge=0.0, description="Total square feet of basement area")
    FirstFlrSF: float = Field(..., alias="1stFlrSF", ge=0.0, description="First floor square feet")
    SecondFlrSF: float = Field(0.0, alias="2ndFlrSF", ge=0.0, description="Second floor square feet")
    GrLivArea: float = Field(..., ge=0.0, description="Above grade living area square feet")
    FullBath: int = Field(..., ge=0, le=6, description="Full bathrooms above grade")
    HalfBath: int = Field(0, ge=0, le=4, description="Half baths above grade")
    BsmtFullBath: int = Field(0, ge=0, le=4, description="Basement full bathrooms")
    BsmtHalfBath: int = Field(0, ge=0, le=4, description="Basement half bathrooms")
    TotRmsAbvGrd: int = Field(..., ge=1, le=20, description="Total rooms above grade (does not include bathrooms)")
    Fireplaces: int = Field(0, ge=0, le=6, description="Number of fireplaces")
    GarageCars: int = Field(..., ge=0, le=6, description="Size of garage in car capacity")
    GarageArea: float = Field(..., ge=0.0, description="Size of garage in square feet")
    LotArea: float = Field(..., ge=0.0, description="Lot size in square feet")

    # Key categorical features
    Neighborhood: str = Field(..., description="Physical location within Ames city limits")
    BldgType: str = Field("1Fam", description="Type of dwelling")
    HouseStyle: str = Field("2Story", description="Style of dwelling")
    ExterQual: str = Field("Gd", description="Exterior material quality")
    KitchenQual: str = Field("Gd", description="Kitchen quality")
    Foundation: str = Field("PConc", description="Type of foundation")
    HeatingQC: str = Field("Ex", description="Heating quality and condition")
    CentralAir: str = Field("Y", description="Central air conditioning (Y/N)")
    SaleCondition: str = Field("Normal", description="Condition of sale")


class ValuationConfidenceInterval(BaseModel):
    lower_bound_usd: float
    upper_bound_usd: float
    confidence_level: str = "95%"


class ValuationPredictionResponse(BaseModel):
    """Output prediction response with appraisal estimate and error bounds."""

    predicted_price_usd: float = Field(..., description="Estimated market sale price")
    confidence_interval: ValuationConfidenceInterval
    model_version: str
    status: str = "success"


class BatchPropertyInput(BaseModel):
    properties: List[PropertyInput]


class BatchValuationResponse(BaseModel):
    predictions: List[ValuationPredictionResponse]
    total_processed: int
