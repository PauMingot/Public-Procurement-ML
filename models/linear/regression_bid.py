"""Linear regression models (Ridge, Lasso, ElasticNet) for tender bid discount prediction.

Predicts percentage discount directly with L1/L2 penalties and converts results
to euros for transparent performance evaluation against the amount formulation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    BUDGET_COL,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_BID_PCT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.data_cleaning import clean_dataframe
from src.metrics import evaluate_bid_in_euros, mdape
from src.visualization import plot_boxplot_ape, plot_scatter_real_vs_pred

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_regression_bid() -> None:
    """Train and evaluate linear models for bid discount percentage."""
    out_dir = OUTPUT_DIR / "linear_bid"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets from %s...", TRAIN_PATH.parent)
    try:
        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        test_df = pd.read_parquet(TEST_PATH)
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    train_df = clean_dataframe(train_df, target="bid_pct")
    val_df = clean_dataframe(val_df, target="bid_pct")
    test_df = clean_dataframe(test_df, target="bid_pct")

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    test_fold = np.concatenate([
        np.full(len(train_df), -1),
        np.zeros(len(val_df)),
    ])
    ps = PredefinedSplit(test_fold)

    y_train_pct = full_train_df[TARGET_BID_PCT].values
    y_test_pct = test_df[TARGET_BID_PCT].values
    budget_test = test_df[BUDGET_COL].values
    y_test_euros_real = budget_test * (1.0 - y_test_pct / 100.0)

    # 1. Baseline: 0% discount
    metrics_baseline = evaluate_bid_in_euros(
        "Trivial Baseline (0% discount)",
        y_test_euros_real,
        np.zeros_like(y_test_pct),
        budget_test,
    )

    # Features selection
    num_features = [
        BUDGET_COL,
        "descuento_medio_hist",
        "duracion_proceso_dias",
        "dias_desde_ultima_licitacion",
        "valor_estimado_imputado",
        "n_licitaciones_hist",
    ]
    cat_features = ["sistema_contratacion", "tipo_procedimiento", "com_aut_adjudicador"]

    num_features = [c for c in num_features if c in full_train_df.columns]
    cat_features = [c for c in cat_features if c in full_train_df.columns]

    X_train_m = full_train_df[num_features + cat_features].copy()
    X_test_m = test_df[num_features + cat_features].copy()

    for col in num_features:
        X_train_m[f"log_{col}"] = np.log1p(X_train_m[col].clip(lower=0))
        X_test_m[f"log_{col}"] = np.log1p(X_test_m[col].clip(lower=0))
    X_train_m.drop(columns=num_features, inplace=True)
    X_test_m.drop(columns=num_features, inplace=True)

    log_num_cols = [c for c in X_train_m.columns if c.startswith("log_")]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), log_num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), cat_features),
    ], remainder="passthrough")

    pipe_ridge = Pipeline([("preprocessor", preprocessor), ("model", Ridge(random_state=RANDOM_STATE))])
    pipe_lasso = Pipeline([("preprocessor", preprocessor), ("model", Lasso(random_state=RANDOM_STATE, max_iter=10000))])
    pipe_elastic = Pipeline([("preprocessor", preprocessor), ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=10000))])

    logger.info("Optimizing Ridge regression for bid discount...")
    grid_ridge = GridSearchCV(pipe_ridge, {"model__alpha": [0.1, 1.0, 10.0, 100.0]}, cv=ps, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train_m, y_train_pct)
    best_ridge = grid_ridge.best_estimator_

    logger.info("Optimizing Lasso regression for bid discount...")
    grid_lasso = GridSearchCV(pipe_lasso, {"model__alpha": [0.001, 0.01, 0.1, 1.0]}, cv=ps, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train_m, y_train_pct)
    best_lasso = grid_lasso.best_estimator_

    logger.info("Optimizing ElasticNet regression for bid discount...")
    grid_elastic = GridSearchCV(pipe_elastic, {"model__alpha": [0.001, 0.01, 0.1], "model__l1_ratio": [0.2, 0.5, 0.8]}, cv=ps, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train_m, y_train_pct)
    best_elastic = grid_elastic.best_estimator_

    metrics_ridge = evaluate_bid_in_euros("Ridge Regression (% Bid -> €)", y_test_euros_real, best_ridge.predict(X_test_m), budget_test)
    metrics_lasso = evaluate_bid_in_euros("Lasso Regression (% Bid -> €)", y_test_euros_real, best_lasso.predict(X_test_m), budget_test)
    metrics_elastic = evaluate_bid_in_euros("ElasticNet Regression (% Bid -> €)", y_test_euros_real, best_elastic.predict(X_test_m), budget_test)

    df_comparison = pd.DataFrame([metrics_baseline, metrics_ridge, metrics_lasso, metrics_elastic])
    logger.info("Linear Bid Models Summary:\n%s", df_comparison.to_markdown(index=False, floatfmt=",.4f"))
    df_comparison.to_csv(out_dir / "linear_bid_comparison.csv", index=False)

    # Diagnostic plots
    y_pred_ridge_pct = best_ridge.predict(X_test_m)
    y_pred_ridge_euros = budget_test * (1.0 - np.clip(y_pred_ridge_pct, 0.0, 100.0) / 100.0)

    plot_scatter_real_vs_pred(
        y_test_euros_real,
        y_pred_ridge_euros,
        "Ridge (% Bid Formulation): Real vs Predicted Award Amount",
        out_dir / "scatter_ridge_bid_euros.png",
        color="#4C72B0",
    )

    # Save model artifact
    joblib.dump(best_ridge, out_dir / "ridge_bid_model.joblib")
    logger.info("Best model saved to: %s", out_dir / "ridge_bid_model.joblib")


if __name__ == "__main__":
    run_regression_bid()
