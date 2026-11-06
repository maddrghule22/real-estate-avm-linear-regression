"""Model validation diagnostics, learning curve generation, and business interpretation."""

import json
import logging
import os
from typing import Dict, List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_learning_curves(
    model_pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: str = "reports/figures",
) -> None:
    """Generate and save learning curves to diagnose bias vs. variance."""
    os.makedirs(output_dir, exist_ok=True)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model_pipeline,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    # Invert negative RMSE
    train_rmse = -np.mean(train_scores, axis=1)
    val_rmse = -np.mean(val_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    val_std = np.std(-val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, "o-", color="#1f77b4", label="Training RMSE")
    plt.fill_between(train_sizes, train_rmse - train_std, train_rmse + train_std, alpha=0.15, color="#1f77b4")
    plt.plot(train_sizes, val_rmse, "s-", color="#d62728", label="Cross-Validation RMSE")
    plt.fill_between(train_sizes, val_rmse - val_std, val_rmse + val_std, alpha=0.15, color="#d62728")

    plt.title("Learning Curves: Best Production Model (Ridge Regression)", fontsize=12, fontweight="bold")
    plt.xlabel("Training Set Size (Number of Properties)")
    plt.ylabel("RMSE ($)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "learning_curve.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Learning curves saved to {fig_path}")


def analyze_residuals_by_segment(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    df_test: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> Dict[str, Dict[str, float]]:
    """Segment residual errors by price bracket and property vintage to detect systematic biases."""
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_test - y_pred
    pct_errors = np.abs(residuals / y_test) * 100.0

    eval_df = df_test.copy()
    eval_df["ActualPrice"] = y_test
    eval_df["PredictedPrice"] = y_pred
    eval_df["Residual"] = residuals
    eval_df["PctError"] = pct_errors

    # 1. Price Tier Segmentation
    eval_df["PriceTier"] = pd.qcut(
        eval_df["ActualPrice"],
        q=3,
        labels=["Affordable (<$135k)", "Mid-Market ($135k-$205k)", "Premium (>$205k)"],
    )

    segment_metrics = {}
    for tier, group in eval_df.groupby("PriceTier", observed=True):
        tier_rmse = float(np.sqrt(np.mean(group["Residual"] ** 2)))
        tier_mae = float(np.mean(np.abs(group["Residual"])))
        tier_mape = float(np.mean(group["PctError"]))
        segment_metrics[str(tier)] = {
            "count": int(len(group)),
            "rmse": round(tier_rmse, 2),
            "mae": round(tier_mae, 2),
            "mape_pct": round(tier_mape, 2),
        }

    # Plot Segment MAPE
    plt.figure(figsize=(9, 5))
    tiers = list(segment_metrics.keys())
    mapes = [segment_metrics[t]["mape_pct"] for t in tiers]
    bars = plt.bar(tiers, mapes, color=["#3182bd", "#6baed6", "#9ecae1"], edgecolor="black")
    plt.ylabel("Mean Absolute Percentage Error (MAPE %)")
    plt.title("Valuation Accuracy (MAPE %) Across Price Tiers", fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, max(mapes) + 3)
    plt.tight_layout()
    seg_fig_path = os.path.join(output_dir, "segment_residual_analysis.png")
    plt.savefig(seg_fig_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Segment residual plot saved to {seg_fig_path}")

    return segment_metrics


def extract_business_interpretations(
    pipeline,
    output_path: str = "reports/business_interpretation.json",
) -> List[Dict[str, object]]:
    """Extract, rank, and explain Ridge regression coefficients in non-technical business terms."""
    prep_step = pipeline.named_steps["prep"].named_steps["preprocessor"]
    feature_names = prep_step.get_feature_names_out()

    # The regressor is wrapped in TransformedTargetRegressor
    reg = pipeline.named_steps["regressor"].regressor_
    coefs = reg.coef_

    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df["AbsImportance"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by="AbsImportance", ascending=False).reset_index(drop=True)

    interpretations = []
    for _, row in coef_df.head(10).iterrows():
        feat = row["Feature"]
        val = row["Coefficient"]
        # In semi-log models (log(y) = beta * x), a 1-std increase in standardized x implies ~ (exp(beta) - 1) * 100% change in price
        pct_change = (np.exp(val) - 1) * 100.0
        direction = "increases" if val > 0 else "decreases"
        explanation = (
            f"A 1-standard-deviation increase in '{feat}' {direction} expected home market valuation "
            f"by approximately {abs(pct_change):.2f}% (ceteris paribus)."
        )
        interpretations.append(
            {
                "feature": feat,
                "standardized_coefficient": round(float(val), 4),
                "approx_price_impact_pct": round(float(pct_change), 2),
                "business_takeaway": explanation,
            }
        )

    with open(output_path, "w") as f:
        json.dump(interpretations, f, indent=2)

    logger.info(f"Business coefficient interpretations written to {output_path}")
    return interpretations


if __name__ == "__main__":
    best_pipe = joblib.load("models/best_linear_pipeline.joblib")
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train = train_df.drop(columns=["SalePrice"])
    y_train = train_df["SalePrice"].values
    X_test = test_df.drop(columns=["SalePrice"])
    y_test = test_df["SalePrice"].values
    y_pred = best_pipe.predict(X_test)

    plot_learning_curves(best_pipe, X_train, y_train)
    seg_results = analyze_residuals_by_segment(y_test, y_pred, test_df)
    print("\nSegment Residual Performance:")
    print(json.dumps(seg_results, indent=2))

    interp = extract_business_interpretations(best_pipe)
    print("\nTop 5 Drivers of Home Value:")
    for item in interp[:5]:
        print(f" - {item['feature']}: {item['approx_price_impact_pct']}% impact -> {item['business_takeaway']}")
