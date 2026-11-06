"""Exploratory Data Analysis (EDA) module for Ames Housing Dataset."""

import json
import logging
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Style settings for publication-ready figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150


def analyze_target_variable(df: pd.DataFrame, target_col: str = "SalePrice", output_dir: str = "reports/figures") -> Dict[str, float]:
    """Analyze target variable distribution and skewness before and after log transform."""
    os.makedirs(output_dir, exist_ok=True)
    target = df[target_col].dropna()
    log_target = np.log1p(target)

    raw_skew = float(target.skew())
    raw_kurt = float(target.kurt())
    log_skew = float(log_target.skew())
    log_kurt = float(log_target.kurt())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw target
    sns.histplot(target, kde=True, ax=axes[0], color="#1f77b4", bins=40, stat="density")
    mu, std = stats.norm.fit(target)
    x = np.linspace(target.min(), target.max(), 100)
    axes[0].plot(x, stats.norm.pdf(x, mu, std), "r--", lw=2, label="Normal fit")
    axes[0].set_title(f"Raw SalePrice\nSkewness: {raw_skew:.2f} | Kurtosis: {raw_kurt:.2f}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sale Price ($)")
    axes[0].legend()

    # Log1p target
    sns.histplot(log_target, kde=True, ax=axes[1], color="#2ca02c", bins=40, stat="density")
    mu_log, std_log = stats.norm.fit(log_target)
    x_log = np.linspace(log_target.min(), log_target.max(), 100)
    axes[1].plot(x_log, stats.norm.pdf(x_log, mu_log, std_log), "r--", lw=2, label="Normal fit")
    axes[1].set_title(f"Log1p(SalePrice)\nSkewness: {log_skew:.2f} | Kurtosis: {log_kurt:.2f}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Log1p(SalePrice)")
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "target_distribution.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Target distribution plot saved to: {fig_path}")

    return {
        "raw_skewness": round(raw_skew, 3),
        "raw_kurtosis": round(raw_kurt, 3),
        "log_skewness": round(log_skew, 3),
        "log_kurtosis": round(log_kurt, 3),
    }


def analyze_missing_values(df: pd.DataFrame, output_dir: str = "reports/figures") -> pd.DataFrame:
    """Detect and quantify missing values across features."""
    os.makedirs(output_dir, exist_ok=True)
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100.0
    missing_df = pd.DataFrame({"MissingCount": null_counts, "MissingPct": null_pct})
    missing_df = missing_df[missing_df["MissingCount"] > 0].sort_values(by="MissingPct", ascending=False)

    if not missing_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_df.head(15).index, y=missing_df.head(15)["MissingPct"], palette="Blues_r")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("% Missing Values")
        plt.title("Top 15 Features by Missing Percentage", fontsize=12, fontweight="bold")
        plt.axhline(50, color="red", linestyle="--", alpha=0.7, label="50% Threshold")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "missing_data_bar.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Missing data plot saved to: {fig_path}")

    return missing_df


def analyze_correlations(
    df: pd.DataFrame, target_col: str = "SalePrice", top_n: int = 12, output_dir: str = "reports/figures"
) -> List[Tuple[str, float]]:
    """Compute Pearson correlation with target and plot collinearity matrix."""
    os.makedirs(output_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    corrs = numeric_df.corr()[target_col].sort_values(ascending=False)

    top_corr_features = corrs.head(top_n).index.tolist()
    corr_submatrix = numeric_df[top_corr_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_submatrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson Correlation"},
    )
    plt.title(f"Correlation Heatmap: Top {top_n} Features Correlated with {target_col}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Correlation heatmap saved to: {fig_path}")

    return [(idx, round(float(val), 3)) for idx, val in corrs.head(top_n).items()]


def detect_outliers(
    df: pd.DataFrame,
    feature: str = "GrLivArea",
    target: str = "SalePrice",
    output_dir: str = "reports/figures",
) -> List[int]:
    """Identify influential outliers via visual scatter and domain rule (De Cock guidelines)."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    # Flag potential high-leverage outliers: Living Area > 4000 sqft with low price (< 300,000)
    outlier_mask = (df[feature] > 4000) & (df[target] < 300000)
    outlier_indices = df[outlier_mask].index.tolist()

    sns.scatterplot(data=df, x=feature, y=target, alpha=0.6, color="#1f77b4", label="Normal Transactions")
    if outlier_indices:
        sns.scatterplot(
            data=df.loc[outlier_indices],
            x=feature,
            y=target,
            color="red",
            s=120,
            marker="X",
            label="High-Leverage Outliers (>4000 sqft, <$300k)",
        )

    plt.title(f"Outlier Diagnostic: {feature} vs {target}", fontsize=12, fontweight="bold")
    plt.xlabel(f"{feature} (Above Grade Living Area SqFt)")
    plt.ylabel(f"{target} ($)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "outlier_scatter.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Outlier scatter plot saved to: {fig_path}")

    return outlier_indices


def run_full_eda(data_path: str = "data/raw/ames_housing.csv") -> Dict[str, object]:
    """Run full exploratory data analysis pipeline."""
    df = pd.read_csv(data_path)
    target_stats = analyze_target_variable(df)
    missing_df = analyze_missing_values(df)
    correlations = analyze_correlations(df)
    outliers = detect_outliers(df)

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "target_stats": target_stats,
        "missing_features_count": int((missing_df["MissingCount"] > 0).sum()),
        "top_missing_features": missing_df.head(10).to_dict()["MissingPct"],
        "top_correlations": dict(correlations),
        "identified_outlier_indices": outliers,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("EDA completed. Summary written to reports/eda_summary.json")
    return summary


if __name__ == "__main__":
    summary = run_full_eda()
    print(json.dumps(summary, indent=2))
