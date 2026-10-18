"""Baseline models and Decision Tree (CART) for tender bid discount prediction.

Predicts percentage discount (0-100%) and translates predictions back to
euros using the tender budget for direct comparison against amount models.
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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from config import (
    BUDGET_COL,
    EXTRA_LEAKAGE_BID,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_AMOUNT,
    TARGET_BID_PCT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    get_dataset_path,
)
from src.data_cleaning import clean_dataframe, drop_leakage_columns
from src.metrics import evaluate_bid_in_euros, mdape
from src.preprocessing import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_baseline_bid() -> None:
    """Train and evaluate baseline and CART models for percentage bid discount."""
    out_dir = OUTPUT_DIR / "baselines_bid"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_df = pd.read_parquet(get_dataset_path("train"))
        val_df = pd.read_parquet(get_dataset_path("val"))
        test_df = pd.read_parquet(get_dataset_path("test"))
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    # Clean data and compute TARGET_BID_PCT
    train_df = clean_dataframe(train_df, target="bid_pct")
    val_df = clean_dataframe(val_df, target="bid_pct")
    test_df = clean_dataframe(test_df, target="bid_pct")

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Save target and budget arrays for euro conversion
    y_train = full_train_df[TARGET_BID_PCT].copy()
    y_test_pct = test_df[TARGET_BID_PCT].values
    budget_test = test_df[BUDGET_COL].values
    y_test_euros_real = budget_test * (1.0 - y_test_pct / 100.0)

    # Drop leakage columns
    drop_cols = LEAKAGE_COLS + EXTRA_LEAKAGE_BID
    full_train_df = drop_leakage_columns(full_train_df, drop_cols)
    test_df = drop_leakage_columns(test_df, drop_cols)

    X_train = full_train_df.drop(columns=[TARGET_BID_PCT], errors="ignore")
    X_test = test_df.drop(columns=[TARGET_BID_PCT], errors="ignore")

    # 1. Trivial Baseline: predict 0% discount (Award = Budget)
    metrics_baseline = evaluate_bid_in_euros(
        "Trivial Baseline (0% discount)",
        y_test_euros_real,
        np.zeros_like(y_test_pct),
        budget_test,
    )

    # 2. CART Decision Tree Pipeline
    logger.info("Configuring CART pipeline for percentage discount...")
    preprocessor = build_preprocessor(X_train)

    pipeline_cart = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "model__max_depth": [5, 10, 15, 20],
        "model__min_samples_leaf": [10, 25, 50, 100],
    }

    grid_search = GridSearchCV(
        estimator=pipeline_cart,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    logger.info("Fitting CART GridSearchCV...")
    grid_search.fit(X_train, y_train)
    best_cart = grid_search.best_estimator_
    logger.info("Best CART params: %s", grid_search.best_params_)

    # Evaluation on Test set
    y_pred_cart_pct = best_cart.predict(X_test)
    metrics_cart = evaluate_bid_in_euros(
        "CART Decision Tree (% Bid -> €)",
        y_test_euros_real,
        y_pred_cart_pct,
        budget_test,
    )

    df_results = pd.DataFrame([metrics_baseline, metrics_cart])
    logger.info("Test Results in Real Euros:\n%s", df_results.to_markdown(index=False, floatfmt=",.4f"))
    df_results.to_csv(out_dir / "baseline_cart_bid_results.csv", index=False)

    # Save model artifact
    model_path = out_dir / "cart_bid_model.joblib"
    joblib.dump(best_cart, model_path)
    logger.info("Model saved to: %s", model_path)


if __name__ == "__main__":
    run_baseline_bid()
