"""Baseline models and Decision Tree (CART) for tender award amount prediction.

Evaluates:
1. Naive baseline: predicting that award price equals the base budget.
2. CART (DecisionTreeRegressor) with hyperparameter search (GridSearchCV) in log space.
3. Conversion of predictions back to euros with R², MAE, and MdAPE metrics.
"""

from __future__ import annotations

import logging
import sys
from math import sqrt
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from config import (
    BUDGET_COL,
    EXTRA_LEAKAGE_AMOUNT,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_AMOUNT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.data_cleaning import clean_dataframe, drop_leakage_columns
from src.metrics import evaluate_in_euros, mdape
from src.preprocessing import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_baseline_amount() -> None:
    """Train and evaluate baseline and CART models for award amount."""
    out_dir = OUTPUT_DIR / "baselines_amount"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading train, val, test datasets from %s...", TRAIN_PATH.parent)
    try:
        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        test_df = pd.read_parquet(TEST_PATH)
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    # Data cleaning
    train_df = clean_dataframe(train_df, target="amount")
    val_df = clean_dataframe(val_df, target="amount")
    test_df = clean_dataframe(test_df, target="amount")

    # Combine train + val for standard evaluation
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Prevent data leakage
    drop_cols = LEAKAGE_COLS + EXTRA_LEAKAGE_AMOUNT
    full_train_df = drop_leakage_columns(full_train_df, drop_cols)
    test_df = drop_leakage_columns(test_df, drop_cols)

    # Feature matrices and targets
    X_train = full_train_df.drop(columns=[TARGET_AMOUNT], errors="ignore")
    y_train = full_train_df[TARGET_AMOUNT].copy()
    y_train_log = np.log1p(y_train)

    X_test = test_df.drop(columns=[TARGET_AMOUNT], errors="ignore")
    y_test_euros = test_df[TARGET_AMOUNT].values
    presupuesto_test = test_df[BUDGET_COL].values

    # 1. Trivial Baseline: Award = Base Budget
    logger.info("Evaluating Model 0: Trivial Baseline (Award = Base Budget)...")
    metrics_baseline = {
        "Model": "Trivial Baseline (Budget = Award)",
        "R²": r2_score(y_test_euros, presupuesto_test),
        "MAE (€)": mean_absolute_error(y_test_euros, presupuesto_test),
        "MdAPE (%)": mdape(y_test_euros, presupuesto_test),
    }

    # 2. Decision Tree (CART) Pipeline
    logger.info("Configuring CART pipeline with GridSearchCV...")
    preprocessor = build_preprocessor(X_train)

    pipeline_cart = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "model__max_depth": [5, 10, 15],
        "model__min_samples_leaf": [25, 50, 100],
    }

    grid_search = GridSearchCV(
        estimator=pipeline_cart,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    logger.info("Training CART grid search...")
    grid_search.fit(X_train, y_train_log)
    best_cart = grid_search.best_estimator_
    logger.info("Best CART parameters: %s", grid_search.best_params_)

    # Evaluation on Test set
    y_pred_log_cart = best_cart.predict(X_test)
    metrics_cart = evaluate_in_euros("CART Decision Tree (Optimized)", y_test_euros, y_pred_log_cart)

    # Results table
    df_results = pd.DataFrame([metrics_baseline, metrics_cart])
    logger.info("Evaluation results on Test set:\n%s", df_results.to_markdown(index=False, floatfmt=",.4f"))
    df_results.to_csv(out_dir / "baseline_cart_amount_results.csv", index=False)

    # Save model artifact
    model_path = out_dir / "cart_amount_model.joblib"
    joblib.dump(best_cart, model_path)
    logger.info("Model saved to: %s", model_path)


if __name__ == "__main__":
    run_baseline_amount()
