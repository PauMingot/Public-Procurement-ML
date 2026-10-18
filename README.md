# 🏛️ Public Procurement ML: Tender Price & Bid Discount Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Framework: Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)

An end-to-end Machine Learning pipeline and interactive auction simulator for predicting award amounts and bid discount percentages in **Spanish Public Procurement** (*Plataforma de Contratación del Sector Público - PLACE*).

---

## 📌 Executive Summary & Business Motivation

Public procurement accounts for approximately **10–15% of Spain's GDP**. In competitive tenders, public administrations must benchmark fair prices and detect disproportionate price reductions (*bajas temerarias*), while contractors require data-driven pricing intelligence to submit winning bids without sacrificing margin.

This project addresses these challenges through a comparative study of **two complementary problem formulations**:
1. **Direct Award Price Prediction**: Modelling the absolute award amount ($\text{EUR}$) in log space ($\log(1 + y)$).
2. **Relative Bid Discount Prediction**: Directly modelling the percentage discount $\text{pct\_baja} = \frac{\text{Budget} - \text{Award}}{\text{Budget}} \times 100 \in [0, 100]\%$ and converting back to euros for an *apples-to-apples* evaluation.

---

## 🏆 Model Benchmark & Empirical Results

All models were evaluated on an independent, temporally segregated holdout test set. Results are converted to **real euros** to ensure fair comparison:

| Model Architecture | Target Formulation | $R^2$ Score | MAE (€) | MdAPE (%) | Key Characteristics |
|:---|:---|:---:|:---:|:---:|:---|
| **Trivial Baseline** | Copy Base Budget | $\approx 0.89$ | Base reference | Base reference | Assumes zero discount |
| **Simple Linear (Log-Log)** | Award Amount | $0.941$ | Significantly lower | $14.2\%$ | Univariate budget benchmark |
| **Lasso Regression (L1)** | Award Amount | $0.958$ | $-32\%$ vs Baseline | $11.8\%$ | Feature selection via L1 sparsity |
| **Ridge Regression (L2)** | % Bid Discount | $0.962$ | $-36\%$ vs Baseline | $10.9\%$ | Penalized multivariant discount model |
| **CART Decision Tree** | Award / % Bid | $0.971$ | $-44\%$ vs Baseline | $8.4\%$ | Non-linear splits with GridSearchCV |
| **Random Forest (Winner)** | Award Amount | **$0.984$** | **$-62\%$ vs Baseline** | **$4.1\%$** | Ensembling 300 trees (Breiman limit) |
| **XGBoost (Winner)** | % Bid Discount | **$0.986$** | **$-65\%$ vs Baseline** | **$3.8\%$** | Gradient boosting with $L_2$ regularization ($\lambda=40$) |

> **Key Takeaway**: Ensemble methods (Random Forest and XGBoost) outperformed linear baselines, lowering the **Median Absolute Percentage Error (MdAPE)** to below **$4\%$**, proving robust against heavy-tailed contract distributions.

---

## 🛡️ Data Leakage Prevention & Data Quality Controls

A common pitfall in public procurement data science is **data leakage** from post-award variables. This pipeline enforces strict pre-award feature isolation:

- **Strict Elimination of Ex-Post Variables**: Columns like `lote_numero_ofertas_recibidas`, `lote_precio_oferta_mas_baja`, `lote_precio_oferta_mas_alta`, `es_sobrecoste`, and `duracion_proceso_dias` are strictly removed before training.
- **Economic Coherence Rules**:
  - Filter out unsuccessful tenders (`es_exito == 1`).
  - Award amount cannot exceed base budget ($\text{Award} \le \text{Budget}$).
  - Filter out unit-price framework agreements where estimated value exceeds budget by $>10\times$.
  - Discard symbolic bids ($\text{Award} < 2.0\text{ €}$).

---

## 🏗️ Repository Architecture

The project adheres to modular, reproducible production standards:

```
Public-Procurement-ML/
├── .gitignore                      # Excludes raw data, serialized models & caches
├── requirements.txt                # Pinned dependencies for full reproducibility
├── config.py                       # Central configuration (paths, column schemas, hyperparameters)
│
├── src/                            # Reusable core modules
│   ├── __init__.py
│   ├── data_cleaning.py            # Business logic filters & data leakage suppression
│   ├── metrics.py                  # MdAPE, euro-scale conversions & evaluation routines
│   ├── preprocessing.py            # Sklearn ColumnTransformer builders
│   ├── feature_importance.py       # One-hot dummy parent feature aggregation
│   └── visualization.py            # Automated, memory-safe plotting routines
│
├── eda/                            # Exploratory Data Analysis
│   ├── eda_amount.py               # Award amount distributions & correlation analysis
│   ├── eda_bid_discount.py         # Bid discount percentage distributions
│   └── eda_companies.py            # Contractor market concentration & specialization
│
├── models/                         # Machine learning model suites
│   ├── baselines/                  # Naive baselines & CART decision trees
│   │   ├── baseline_amount.py
│   │   └── baseline_bid.py
│   ├── linear/                     # Regularized linear models (Ridge, Lasso, ElasticNet)
│   │   ├── regression_amount.py
│   │   └── regression_bid.py
│   ├── random_forest/              # Random Forest training & hyperparameter tuning
│   │   ├── rf_amount.py
│   │   ├── rf_bid.py
│   │   ├── tuning_rf_amount.py
│   │   └── tuning_rf_bid.py
│   └── xgboost/                    # Gradient Boosting training & early stopping tuning
│       ├── xgb_amount.py
│       ├── xgb_bid.py
│       ├── tuning_xgb_amount.py
│       └── tuning_xgb_bid.py
│
├── app/                            # Interactive procurement simulator
│   └── streamlit_app.py            # Streamlit dashboard for tender simulation
│
└── scripts/                        # Data processing & ETL utilities
    └── build_company_profiles.py   # Extracts lightweight contractor track records
```

---

## 🔍 Model Interpretability & Explainability

All tree-based models integrate **SHAP (SHapley Additive exPlanations)** and aggregated feature importances:
- **Parent Feature Aggregation**: One-hot encoded categorical dummies are re-aggregated to their parent variable to produce clean, interpretable rankings.
- **Key Predictors**: Historical contractor discount (`descuento_medio_hist`), base budget, contract type (`tipo_contrato`), contracting authority cluster (`organo_cluster_label`), and regional geography consistently drive predictions.

---

## 🚀 Quickstart & Usage

### 1. Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/PauMingot/Public-Procurement-ML.git
cd Public-Procurement-ML

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set the environment variable `DATA_DIR` if your parquet datasets are stored outside `./data`:

```bash
# Optional: specify custom dataset location
export DATA_DIR="/path/to/procurement_data"  # On Windows: $env:DATA_DIR="C:\path\to\data"
```

### 3. Running Exploratory Data Analysis (EDA)

```bash
python eda/eda_amount.py
python eda/eda_bid_discount.py
python eda/eda_companies.py
```

### 4. Training Models

```bash
# Train Random Forest models
python models/random_forest/rf_amount.py
python models/random_forest/rf_bid.py

# Train XGBoost models
python models/xgboost/xgb_amount.py
python models/xgboost/xgb_bid.py
```

### 5. Running the Streamlit Auction Simulator

```bash
# Generate company profiles for the app
python scripts/build_company_profiles.py

# Launch the simulator
streamlit run app/streamlit_app.py
```

---

## 👤 Author & Contact

**Pau Mingot**  
- GitHub: [@PauMingot](https://github.com/PauMingot)  
- Focus: Data Science, Predictive Modelling & Quantitative Analytics
