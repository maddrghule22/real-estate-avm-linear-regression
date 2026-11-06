"""FastAPI production inference microservice for Automated Valuation Model (AVM)."""

from contextlib import asynccontextmanager
import logging
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
    PropertyInput,
    ValuationPredictionResponse,
    ValuationConfidenceInterval,
    BatchPropertyInput,
    BatchValuationResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("avm_api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_linear_pipeline.joblib")
MODEL_VERSION = "1.0.0-ridge-cv"
# Root Mean Squared Log Error derived from cross-validation on log scale for prediction intervals
RESIDUAL_STD_LOG = 0.125

ml_pipeline = None


def get_pipeline():
    """Lazily or eagerly fetch loaded ML pipeline."""
    global ml_pipeline
    if ml_pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
        logger.info(f"Loading pipeline from {MODEL_PATH}...")
        ml_pipeline = joblib.load(MODEL_PATH)
    return ml_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML model artifact during server startup."""
    get_pipeline()
    yield
    logger.info("Shutting down AVM inference service.")


app = FastAPI(
    title="Residential Automated Valuation Model (AVM) API",
    description="Production-grade Linear Regression appraisal microservice powered by Ridge Regression.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _predict_single_property(prop: PropertyInput) -> ValuationPredictionResponse:
    """Format payload and compute prediction with 95% confidence interval."""
    pipeline = get_pipeline()
    # Convert input payload to DataFrame preserving original column aliases
    data_dict = prop.model_dump(by_alias=True)
    df_in = pd.DataFrame([data_dict])

    pred_dollar = float(pipeline.predict(df_in)[0])

    # 95% Prediction interval: exp(ln(y) +/- 1.96 * se)
    lower_bound = pred_dollar * float(np.exp(-1.96 * RESIDUAL_STD_LOG))
    upper_bound = pred_dollar * float(np.exp(1.96 * RESIDUAL_STD_LOG))

    return ValuationPredictionResponse(
        predicted_price_usd=round(pred_dollar, 2),
        confidence_interval=ValuationConfidenceInterval(
            lower_bound_usd=round(lower_bound, 2),
            upper_bound_usd=round(upper_bound, 2),
            confidence_level="95%",
        ),
        model_version=MODEL_VERSION,
        status="success",
    )


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Residential Automated Valuation Model (AVM) API",
        "version": MODEL_VERSION,
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check probe for container orchestrators (Kubernetes / Docker)."""
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact not initialized")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=ValuationPredictionResponse, tags=["Inference"])
async def predict_property_valuation(property_input: PropertyInput):
    """Estimate single residential property valuation with 95% confidence interval."""
    try:
        return _predict_single_property(property_input)
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchValuationResponse, tags=["Inference"])
async def predict_batch_valuations(batch_input: BatchPropertyInput):
    """Process high-throughput batch property appraisals."""
    try:
        predictions = [_predict_single_property(p) for p in batch_input.properties]
        return BatchValuationResponse(
            predictions=predictions,
            total_processed=len(predictions),
        )
    except Exception as e:
        logger.error(f"Batch inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference execution failed: {str(e)}",
        )


@app.get("/model/metadata", tags=["Model Governance"])
async def get_model_metadata():
    """Retrieve model governance details and baseline validation metrics."""
    return {
        "model_architecture": "Ridge Regression with Log-Transformed Target",
        "hyperparameter_alpha": 5.0,
        "holdout_test_rmse": 24441.97,
        "holdout_test_mae": 15443.99,
        "holdout_test_mape": 9.31,
        "holdout_test_r2": 0.9221,
        "holdout_test_adjusted_r2": 0.8926,
    }
