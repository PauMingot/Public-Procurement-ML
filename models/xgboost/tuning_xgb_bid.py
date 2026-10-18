"""Hyperparameter tuning for XGBoost Regressor (Percentage Bid Discount).

Performs grid search with early stopping on validation loss, evaluating all
candidate configurations in real euros against the tender budget.
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from config import (
    BUDGET_COL,
    EXTRA_LEAKAGE_BID,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_BID_PCT,
    TRAIN_PATH,
    VAL_PATH,
)
from src.data_cleaning import clean_dataframe, drop_leakage_columns
from src.preprocessing import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tuning_xgb_bid() -> None:
    """Execute hyperparameter grid search with early stopping for XGBoost Bid Discount."""
    out_dir = OUTPUT_DIR / "tuning_xgb_bid"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training and validation sets for XGBoost Bid tuning...")
    try:
        train_df = clean_dataframe(pd.read_parquet(TRAIN_PATH), target="bid_pct")
        val_df = clean_dataframe(pd.read_parquet(VAL_PATH), target="bid_pct")
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    budget_val = val_df[BUDGET_COL].values
    y_val_pct = val_df[TARGET_BID_PCT].values
    y_val_euros_real = budget_val * (1.0 - y_val_pct / 100.0)

    drop_cols = LEAKAGE_COLS + EXTRA_LEAKAGE_BID
    train_df = drop_leakage_columns(train_df, drop_cols)
    val_df = drop_leakage_columns(val_df, drop_cols)

    X_train = train_df.drop(columns=[TARGET_BID_PCT], errors="ignore")
    y_train = train_df[TARGET_BID_PCT].values

    X_val = val_df.drop(columns=[TARGET_BID_PCT], errors="ignore")
    y_val = val_df[TARGET_BID_PCT].values

    logger.info("Transforming features...")
    preprocessor = build_preprocessor(X_train)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [10, 15, 20],
        "reg_lambda": [10, 20, 40],
        "gamma": [0, 1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logger.info("Testing %d parameter combinations for %% bid discount...", len(combinations))

    best_mae_euros = float("inf")
    best_params: dict = {}
    best_n_trees = 0
    results_records = []

    for idx, params in enumerate(combinations, start=1):
        xgb = XGBRegressor(
            n_estimators=1500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="mae",
            **params,
        )

        xgb.fit(
            X_train_trans,
            y_train,
            eval_set=[(X_val_trans, y_val)],
            verbose=False,
        )

        pred_val_pct = np.clip(xgb.predict(X_val_trans), 0.0, 100.0)
        pred_val_euros = budget_val * (1.0 - pred_val_pct / 100.0)
        mae_val_euros = mean_absolute_error(y_val_euros_real, pred_val_euros)
        n_trees = xgb.best_iteration

        logger.info("[%d/%d] Trees: %s | Euro Val MAE: %.2f € | %s", idx, len(combinations), n_trees, mae_val_euros, params)
        results_records.append({**params, "best_iteration": n_trees, "val_mae_euros": mae_val_euros})

        if mae_val_euros < best_mae_euros:
            best_mae_euros = mae_val_euros
            best_params = params
            best_n_trees = n_trees

    df_results = pd.DataFrame(results_records).sort_values("val_mae_euros")
    df_results.to_csv(out_dir / "tuning_xgb_bid_results.csv", index=False)

    logger.info("XGBoost Bid Tuning Complete.")
    logger.info("Winner Parameters: %s | Optimal Trees: %d | Best Euro MAE: %.2f €", best_params, best_n_trees, best_mae_euros)


if __name__ == "__main__":
    run_tuning_xgb_bid()
