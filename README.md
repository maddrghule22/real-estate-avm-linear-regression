# Residential Automated Valuation Model (AVM) — End-to-End Linear Regression Engine

[![CI / Model & API Verification](https://github.com/maddrghule22/real-estate-avm-linear-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/maddrghule22/real-estate-avm-linear-regression/actions)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![Pytest](https://img.shields.io/badge/Pytest-16%20Passed-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📌 Executive Summary
An industry-grade, production-engineered Machine Learning valuation engine centered on **Linear Regression and its regularized variants (Ridge, Lasso, ElasticNet)** for single-family residential property pricing. 

Designed to reflect institutional PropTech and mortgage underwriting standards, this project pairs rigorous econometric inference (Gauss-Markov diagnostics, multicollinearity mitigation, heteroscedasticity containment) with a containerized **FastAPI** microservice serving real-time valuations with **95% prediction intervals**.

```text
               ┌────────────────────────────────────────────────────────┐
               │              Ames Housing Raw Dataset                  │
               │           (1,460 Properties, 79 Features)              │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
       Training Set (80%, N=1166)                       Holdout Test (20%, N=292)
       [Train-only outlier pruning]                     [Strict Zero Leakage]
                  │                                               │
                  ▼                                               ▼
       Feature Engineering Pipeline                    Transformer.transform()
       • Domain: TotalSF, Age, Baths                              │
       • Target: log(1 + y)                                       │
       • Imputation & One-Hot Encoding                            │
                  │                                               │
                  ▼                                               │
       5-Fold CV Hyperparameter Tuning                            │
       • OLS (Statsmodels & Scikit-learn)                         │
       • Ridge (L2 Shrinkage)  ◄── [BEST: RMSE $24.4k]            │
       • Lasso (L1 Sparsity)                                      │
       • ElasticNet (L1 + L2)                                     │
       • Random Forest (Non-linear Benchmark)                    │
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                         Holdout Evaluation & Inference
                                          │
                                          ▼
                       Production FastAPI Microservice (Docker)
                       POST /predict ──> $208,450 [CI: $163k - $265k]
```

---

## 🏢 Business Problem & Economic Framing
- **Domain**: PropTech / Automated Valuation Models (AVMs) / Mortgage Underwriting.
- **Problem Statement**: Underwriting platforms and real estate brokerages require instantaneous, explainable property appraisals to formulate competitive cash bids while defending transaction margins.
- **Asymmetric Risk**:
  - *Overvaluation Risk*: Homes linger on the market (inflating Days-on-Market / DOM), signal distress, force consecutive price drops, and tie up capital.
  - *Undervaluation Risk*: Results in direct seller equity erosion, reduced transaction commissions, and adverse selection.
- **Target Variable ($y$)**: `SalePrice` (USD, continuous, positive, right-skewed).

---

## 📊 Empirical Model Comparison Benchmark

All candidate models were tuned via 5-fold cross-validation on the training set and evaluated on the isolated 20% holdout test partition ($N=292$ properties):

| Model Architecture | Hyperparameters | Holdout RMSE ($) | Holdout MAE ($) | Holdout MAPE (%) | Holdout $R^2$ | Adjusted $R^2$ | Inference Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Mean Baseline** | $\bar{y}_{\text{train}}$ | \$87,619.03 | \$62,575.93 | 41.36% | -0.0009 | -0.0009 | < 0.01 ms |
| **Naive Median Baseline** | $\text{median}(y_{\text{train}})$ | \$88,667.17 | \$59,568.25 | 35.97% | -0.0250 | -0.0250 | < 0.01 ms |
| **Random Forest Benchmark** | `n_est=200, max_depth=None` | \$27,149.34 | \$16,410.87 | 9.91% | 0.9039 | 0.8675 | ~ 18.5 ms |
| **Ordinary Least Squares (OLS)** | Unregularized | \$25,003.44 | \$15,578.11 | 9.42% | 0.9185 | 0.8876 | < 0.1 ms |
| **Lasso Regression ($L_1$)** | $\alpha = 0.0001$ | \$24,758.65 | \$15,470.74 | 9.36% | 0.9201 | 0.8898 | < 0.1 ms |
| **ElasticNet ($L_1/L_2$)** | $\alpha = 0.001, \text{l1\_ratio}=0.1$ | \$24,604.79 | \$15,452.14 | 9.34% | 0.9211 | 0.8911 | < 0.1 ms |
| **🏆 Ridge Regression ($L_2$)** | **$\alpha = 5.0$** | **\$24,441.97** | **\$15,443.99** | **9.31%** | **0.9221** | **0.8926** | **< 0.1 ms** |

### 💡 Why Linear Regression Beat Random Forest
1. **Multicollinearity Stabilization**: By applying $L_2$ shrinkage ($\alpha=5.0$), Ridge gracefully shrinks correlated coefficients (`TotalSF`, `GrLivArea`, `Quality_x_TotalSF`) without the high variance of unregularized OLS.
2. **Smooth Extrapolation**: Random Forest is a step-function ensemble that cannot extrapolate beyond the extreme bounding boxes of training leaves. Ridge's continuous hyperplanes model smooth marginal premiums for premium footage.
3. **Latency & Explainability**: Ridge executes in sub-millisecond speeds (< 0.1 ms vs 18.5 ms for 200 trees) and provides legally auditable coefficient weights required by fair-lending regulations.

---

## 🔬 Econometric Assumption Diagnostics (Gauss-Markov Verification)

A linear model is only as valid as its underlying statistical assumptions:

| Assumption | Diagnostic Method | Empirical Result | Mathematical Interpretation & Engineering Remedy |
|---|---|:---:|---|
| **1. Linearity of Expectation** | Residuals vs. Fitted plot | Flat LOESS line ($E[e\|X] \approx 0$) | Relationship between log-price and engineered square footage/age is strictly linear. |
| **2. Independence of Errors** | Durbin-Watson Test | **$d = 1.974$** (Target: $\approx 2.0$) | Residuals exhibit zero first-order autocorrelation ($1.5 < d < 2.5$ pass criterion). |
| **3. Homoscedasticity** | Breusch-Pagan Test & Scale-Location | $LM = 181.76, p = 0.000$ | Raw prices exhibited acute fanning variance (heteroscedasticity). Applying $y \leftarrow \log(1+y)$ normalized residual spread across budget and luxury segments. |
| **4. Normality of Residuals** | Normal Q-Q Plot & Jarque-Bera | Central quantiles linear, minor tail deviations | Central Limit Theorem ($N=1,166$) ensures asymptotic normality for parameter inference. |
| **5. Multicollinearity** | Variance Inflation Factor (VIF) | `Quality_x_TotalSF`: 43.9<br>`TotalSF`: 26.7<br>`OverallQual`: 12.2 | High collinearity identified in interaction terms. Solved by Ridge $L_2$ penalty, which bounds $(X^T X + \lambda I)^{-1}$ conditioning. |

Diagnostic plots are automatically generated and saved in [`reports/figures/linear_regression_assumptions.png`](reports/figures/linear_regression_assumptions.png).

---

## 📈 Non-Technical Business Interpretation (Coefficients in Plain English)

Because the target variable was modeled in log space ($\ln(\text{SalePrice}) = \beta_0 + \beta_1 z_1 + \dots$), each standardized regression coefficient translates directly into a percentage valuation shift via:
$$\%\Delta y \approx \left( e^{\beta_j} - 1 \right) \times 100\%$$

| Predictive Feature | Std. Coefficient ($\beta$) | Price Impact (%) | Business Takeaway for Appraisers |
|---|:---:|:---:|---|
| **`cat__Neighborhood_Crawfor`** | $+0.1139$ | **$+12.06\%$** | Homes in the historic Crawford neighborhood command a 12.1% premium over baseline locations. |
| **`cat__Neighborhood_StoneBr`** | $+0.0988$ | **$+10.39\%$** | Stone Brook properties command a 10.4% location premium. |
| **`num__TotalSF`** | $+0.0867$ | **$+9.06\%$** | A 1-standard-deviation increase in total square footage (~750 sqft) drives a +9.06% price appreciation. |
| **`cat__Neighborhood_MeadowV`** | $-0.0924$ | **$-8.82\%$** | Meadow Village properties trade at an 8.82% discount relative to market median. |
| **`cat__SaleCondition_Abnorml`** | $-0.0733$ | **$-7.07\%$** | Distressed or non-arm's-length transactions trade at an immediate 7.07% haircut. |
| **`num__HouseAge`** | $-0.0734$ | **$-7.08\%$** | Every ~30-year increase in property vintage depreciates structural value by 7.08%. |

---

## 🛠️ Repository Architecture

```text
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated GitHub Actions test pipeline
├── configs/
│   └── config.yaml                # Pipeline hyperparameters and schema definitions
├── data/
│   ├── raw/                       # Immutable raw dataset (ames_housing.csv)
│   └── processed/                 # Train and test splits (parquet)
├── notebooks/
│   └── residential_avm_walkthrough.ipynb # Executable end-to-end walkthrough
├── src/
│   ├── data/
│   │   ├── loader.py              # OpenML ingestion and local caching
│   │   ├── eda.py                 # Automated EDA and figure generator
│   │   └── split.py               # Leak-free splitting with train-only outlier pruning
│   ├── features/
│   │   └── transformers.py        # Domain feature engineering & ColumnTransformer
│   ├── models/
│   │   ├── baseline.py            # Naive Mean and Median benchmarks
│   │   ├── train.py               # 5-fold CV training (OLS, Ridge, Lasso, ElasticNet, RF)
│   │   └── validation_and_interpretation.py # Learning curves, segments, business coefficients
│   └── utils/
│       ├── metrics.py             # RMSE, MAE, MAPE, R2, Adjusted R2
│       ├── diagnostics.py         # VIF, Breusch-Pagan, Durbin-Watson, Q-Q plots
│       └── monitoring.py          # MLOps Population Stability Index (PSI) & KS drift tests
├── tests/
│   ├── test_api.py                # FastAPI integration tests
│   ├── test_baseline.py           # Baseline model tests
│   ├── test_diagnostics.py        # Assumption & VIF tests
│   ├── test_features.py           # Feature transformer & leakage tests
│   ├── test_models.py             # Serialized artifact verification
│   └── test_monitoring.py         # PSI drift detection tests
├── models/
│   └── best_linear_pipeline.joblib# Serialized production pipeline artifact
├── reports/
│   ├── figures/                   # High-res diagnostic visualizations
│   ├── eda_summary.json           # EDA metrics
│   ├── model_comparison.csv       # Benchmark results table
│   └── vif_analysis.csv           # Multicollinearity report
├── app/
│   ├── main.py                    # Production FastAPI application
│   └── schemas.py                 # Pydantic v2 validation contracts
├── Dockerfile                     # Multi-stage production container
├── docker-compose.yml             # Single-command container deployment
├── requirements.txt               # Locked dependencies
└── README.md                      # Comprehensive documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/maddrghule22/real-estate-avm-linear-regression.git
cd real-estate-avm-linear-regression

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Automated Pipeline
```bash
# 1. Fetch raw data & compute baseline
python -m src.data.loader
python -m src.models.baseline

# 2. Run EDA & generate figures
python -m src.data.eda

# 3. Create clean splits
python -m src.data.split

# 4. Train and tune all candidate models
python -m src.models.train

# 5. Run econometric assumption checks & business interpretation
python -m src.utils.diagnostics
python -m src.models.validation_and_interpretation
```

### 3. Run Test Suite
```bash
pytest tests/ -v
# Output: 16 passed in 5.0s
```

---

## 🐳 Docker Deployment & API Usage

### Run with Docker Compose:
```bash
docker compose up --build
```
The API starts on port `8000`. Swagger documentation is available at `http://localhost:8000/docs`.

### Run Locally with Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Sample cURL Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
       "SaleCondition": "Normal"
     }'
```

### Response JSON (Valuation with 95% Confidence Interval):
```json
{
  "predicted_price_usd": 208450.32,
  "confidence_interval": {
    "lower_bound_usd": 163214.28,
    "upper_bound_usd": 266205.11,
    "confidence_level": "95%"
  },
  "model_version": "1.0.0-ridge-cv",
  "status": "success"
}
```

---

## 💼 Resume Bullet Points (Portfolio Impact)

* **Engineered an end-to-end Automated Valuation Model (AVM)** using regularized Ridge Regression ($\alpha=5.0$) across 1,460 residential properties, improving appraisal precision from a naive baseline RMSE of **\$87.6k to \$24.4k** ($R^2 = 0.922$).
* **Eliminated data leakage** by architecting modular scikit-learn `Pipeline` and `ColumnTransformer` workflows, isolating imputation, $z$-score scaling, and train-only leverage outlier filtering ($N=1,166$ train / $N=292$ test).
* **Diagnosed and mitigated econometric violations** through automated Gauss-Markov checks, using Box-Cox/log transforms to counter heteroscedasticity and applying $L_2$ penalization to stabilize severe multicollinear features ($\text{VIF} > 40$).
* **Outperformed non-linear ensembles (Random Forest)** by **\$2,707 in holdout RMSE** (9.31% vs 9.91% MAPE) while cutting inference latency by **99%** (< 0.1 ms vs 18.5 ms) and ensuring compliance with fair-lending explainability mandates.
* **Productionized and containerized the inference pipeline** with **FastAPI** and **Docker**, implementing strict Pydantic v2 contracts, 95% prediction intervals, 16 automated pytest unit/integration tests, and CI/CD verification.

---

## 🎙️ 3-Minute Interview Talk Track

> **Opening (30 sec)**:  
> *"In real estate valuation, an algorithmic appraisal error isn't just an abstract loss metric—it's asymmetric financial risk. Overpricing a home causes it to sit on the market and accumulate stigma discounts, while underpricing forfeits seller equity. To address this, I built an end-to-end Automated Valuation Engine centered on regularized Linear Regression for single-family residential properties."*

> **Methodology & Leakage Prevention (60 sec)**:  
> *"During exploratory analysis, I noticed that the raw sales price had severe positive skewness (1.88) and kurtosis (6.54), which violated homoscedasticity. Furthermore, two properties with over 4,000 square feet had sold under \$300,000, creating high-leverage anomalies. Rather than cleaning data globally, I instituted a strict split-first design: I removed those two leverage points strictly from the training set, engineered domain-specific features like `TotalSF` and property age within a custom scikit-learn transformer, and fit a log-transformed target pipeline to normalize error variances across low and high valuations."*

> **Modeling, Diagnostics & Results (60 sec)**:  
> *"When looking at the features, variance inflation factor analysis revealed severe multicollinearity—for instance, interaction terms like `Quality_x_TotalSF` had VIFs exceeding 40. Ordinary Least Squares became unstable. By tuning a Ridge Regression model with 5-fold cross-validation ($\alpha=5.0$), the L2 shrinkage stabilized parameter variance. On our holdout test set of 292 unseen homes, Ridge achieved an RMSE of \$24,441, an MAE of \$15,443, and an $R^2$ of 0.922. Crucially, Ridge outperformed our 200-tree Random Forest benchmark, which yielded an RMSE of \$27,149. Because real estate follows smooth marginal returns for square footage, Ridge avoided the step-function boundary errors of decision trees while reducing inference latency to under 0.1 milliseconds."*

> **Production Deployment (30 sec)**:  
> *"Finally, I packaged the entire pipeline into a production FastAPI microservice encapsulated in Docker. The API validates payloads with Pydantic v2, returns 95% prediction intervals based on residual standard errors, and is backed by a 16-test pytest suite and GitHub Actions CI. The result is a mathematically sound, fully auditable valuation service ready for institutional underwriting."*

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
