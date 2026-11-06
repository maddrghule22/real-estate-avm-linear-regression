"""Model training, hyperparameter tuning, and model comparison suite."""

import json
import logging
import os
import time
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

from src.features.transformers import build_full_feature_pipeline
from src.utils.metrics import evaluate_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fit_statsmodels_ols(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit Statsmodels OLS for statistical inference, p-values, and hypothesis testing."""
    # Build pipeline with drop_first=True to avoid dummy variable singularity
    prep_pipeline = build_full_feature_pipeline(drop_first=True)
    X_trans = prep_pipeline.fit_transform(X_train_df)
    feature_names = prep_pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Model target in log space for OLS assumption compliance
    log_y = np.log1p(y_train)

    # Add explicit constant intercept for statsmodels
    X_with_const = sm.add_constant(pd.DataFrame(X_trans, columns=feature_names), has_constant="add")
    ols_model = sm.OLS(log_y, X_with_const).fit()

    os.makedirs("reports", exist_ok=True)
    with open("reports/ols_summary.txt", "w") as f:
        f.write(str(ols_model.summary()))

    logger.info("Statsmodels OLS summary written to reports/ols_summary.txt")
    return ols_model


def run_model_experiments(
    train_path: str = "data/processed/train.parquet",
    test_path: str = "data/processed/test.parquet",
    models_dir: str = "models",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train OLS, Ridge, Lasso, ElasticNet, and Random Forest benchmark."""
    os.makedirs(models_dir, exist_ok=True)
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "SalePrice"
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    # 1. Fit Statsmodels OLS for inference
    logger.info("Fitting Statsmodels OLS for statistical inference...")
    _ = fit_statsmodels_ols(X_train, y_train)

    # Prepare scikit-learn cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Base feature pipeline (drop_first=False for regularized models)
    prep_pipeline = build_full_feature_pipeline(drop_first=False)

    candidate_models = {
        "OLS (LinearRegression)": {
            "model": TransformedTargetRegressor(
                regressor=LinearRegression(),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "params": {},
        },
        "Ridge Regression": {
            "model": TransformedTargetRegressor(
                regressor=Ridge(random_state=42),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "params": {
                "regressor__regressor__alpha": [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
            },
        },
        "Lasso Regression": {
            "model": TransformedTargetRegressor(
                regressor=Lasso(random_state=42, max_iter=10000),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "params": {
                "regressor__regressor__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
            },
        },
        "ElasticNet": {
            "model": TransformedTargetRegressor(
                regressor=ElasticNet(random_state=42, max_iter=10000),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "params": {
                "regressor__regressor__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05],
                "regressor__regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
        "Random Forest (Non-linear Benchmark)": {
            "model": TransformedTargetRegressor(
                regressor=RandomForestRegressor(random_state=42, n_jobs=-1),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "params": {
                "regressor__regressor__n_estimators": [100, 200],
                "regressor__regressor__max_depth": [10, 20, None],
            },
        },
    }

    results = []
    trained_pipelines = {}

    for name, config in candidate_models.items():
        logger.info(f"Training and tuning: {name}...")
        full_pipe = Pipeline(
            steps=[
                ("prep", prep_pipeline),
                ("regressor", config["model"]),
            ]
        )

        start_time = time.time()
        if config["params"]:
            grid = GridSearchCV(
                estimator=full_pipe,
                param_grid=config["params"],
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best_pipe = grid.best_estimator_
            best_params = grid.best_params_
        else:
            full_pipe.fit(X_train, y_train)
            best_pipe = full_pipe
            best_params = "Default"

        fit_duration = time.time() - start_time

        # Test set evaluation
        y_pred = best_pipe.predict(X_test)
        p = X_train.shape[1]
        metrics = evaluate_regression(y_test, y_pred, p=p)

        metrics["Model"] = name
        metrics["Best_Params"] = str(best_params)
        metrics["Train_Time_Sec"] = round(fit_duration, 3)
        results.append(metrics)
        trained_pipelines[name] = best_pipe

        # Save artifacts
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        artifact_path = os.path.join(models_dir, f"{clean_name}.joblib")
        joblib.dump(best_pipe, artifact_path)

    # Compile results table
    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model", "rmse", "mae", "mape_pct", "r2", "adjusted_r2", "Train_Time_Sec", "Best_Params"]
    ].sort_values(by="rmse", ascending=True)

    results_csv = "reports/model_comparison.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Model comparison saved to {results_csv}")

    # Export best linear model as standard production artifact
    best_linear_model = trained_pipelines["Ridge Regression"]
    joblib.dump(best_linear_model, os.path.join(models_dir, "best_linear_pipeline.joblib"))
    logger.info("Saved best production linear model to models/best_linear_pipeline.joblib")

    return results_df, trained_pipelines


if __name__ == "__main__":
    df_res, _ = run_model_experiments()
    print("\n" + "=" * 90)
    print("EMPIRICAL MODEL EVALUATION BENCHMARK ON HOLDOUT TEST SET (N=292)")
    print("=" * 90)
    print(df_res.to_string(index=False))
