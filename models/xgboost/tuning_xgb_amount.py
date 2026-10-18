"""Hyperparameter tuning for XGBoost Regressor (Tender Award Amount).

Explores learning rates, maximum tree depths, and L2 regularization penalties
with early stopping against the validation set evaluated in euros.
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
    EXTRA_LEAKAGE_AMOUNT,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_AMOUNT,
    TRAIN_PATH,
    VAL_PATH,
)
from src.data_cleaning import clean_dataframe, drop_leakage_columns
from src.preprocessing import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tuning_xgb_amount() -> None:
    """Execute hyperparameter grid search with early stopping for XGBoost Amount."""
    out_dir = OUTPUT_DIR / "tuning_xgb_amount"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training and validation sets for XGBoost Amount tuning...")
    try:
        train_df = clean_dataframe(pd.read_parquet(TRAIN_PATH), target="amount")
        val_df = clean_dataframe(pd.read_parquet(VAL_PATH), target="amount")
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    drop_cols = LEAKAGE_COLS + EXTRA_LEAKAGE_AMOUNT
    train_df = drop_leakage_columns(train_df, drop_cols)
    val_df = drop_leakage_columns(val_df, drop_cols)

    X_train = train_df.drop(columns=[TARGET_AMOUNT], errors="ignore")
    y_train_log = np.log1p(train_df[TARGET_AMOUNT])

    X_val = val_df.drop(columns=[TARGET_AMOUNT], errors="ignore")
    y_val_log = np.log1p(val_df[TARGET_AMOUNT])
    y_val_euros_real = val_df[TARGET_AMOUNT].values

    logger.info("Fitting and transforming feature preprocessor...")
    preprocessor = build_preprocessor(X_train)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    param_grid = {
        "learning_rate": [0.1, 0.15, 0.2],
        "max_depth": [2, 4, 6],
        "reg_lambda": [1, 10],
        "gamma": [0],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logger.info("Exploring %d parameter combinations with early stopping (50 rounds)...", len(combinations))

    best_mae = float("inf")
    best_params: dict = {}
    best_n_trees = 0
    max_train_log = float(y_train_log.max())
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
            y_train_log,
            eval_set=[(X_val_trans, y_val_log)],
            verbose=False,
        )

        # Bounded prediction preventing extrapolation beyond historical maximum
        pred_val_log = np.clip(xgb.predict(X_val_trans), 0, max_train_log)
        pred_val_euros = np.expm1(pred_val_log)
        mae_val = mean_absolute_error(y_val_euros_real, pred_val_euros)
        n_trees = xgb.best_iteration

        logger.info("[%d/%d] Trees: %s | Val MAE: %.2f € | %s", idx, len(combinations), n_trees, mae_val, params)
        results_records.append({**params, "best_iteration": n_trees, "val_mae_euros": mae_val})

        if mae_val < best_mae:
            best_mae = mae_val
            best_params = params
            best_n_trees = n_trees

    df_results = pd.DataFrame(results_records).sort_values("val_mae_euros")
    df_results.to_csv(out_dir / "tuning_xgb_amount_results.csv", index=False)

    logger.info("XGBoost Amount Tuning Complete.")
    logger.info("Winner Parameters: %s | Optimal Trees: %d | Best MAE: %.2f €", best_params, best_n_trees, best_mae)


if __name__ == "__main__":
    run_tuning_xgb_amount()
