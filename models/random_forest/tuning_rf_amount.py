"""Hyperparameter tuning and convergence analysis for Random Forest (Award Amount).

Phase 1: Grid Search over tree architecture (max_depth, min_samples_leaf, max_features).
Phase 2: Ensemble size convergence curve over n_estimators (Breiman's error bound).
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
from src.visualization import plot_convergence_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tuning_rf_amount() -> None:
    """Execute two-phase hyperparameter optimization for Random Forest."""
    out_dir = OUTPUT_DIR / "tuning_rf_amount"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training and validation sets for RF Amount tuning...")
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
    y_val_euros_real = val_df[TARGET_AMOUNT].values

    # Pre-transform features once for efficiency
    logger.info("Fitting and transforming feature preprocessor...")
    preprocessor = build_preprocessor(X_train)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    # Phase 1: Architecture Grid
    logger.info("Phase 1: Exploring tree architecture grid...")
    param_grid = {
        "max_depth": [10, 15, 20],
        "min_samples_leaf": [2, 4, 8],
        "max_features": [0.5, 0.8, "sqrt"],
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logger.info("Testing %d parameter combinations with fixed n_estimators=100...", len(combinations))

    best_mae = float("inf")
    best_params: dict = {}

    for idx, params in enumerate(combinations, start=1):
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        rf.fit(X_train_trans, y_train_log)

        pred_val_log = np.clip(rf.predict(X_val_trans), -np.inf, 30)
        pred_val_euros = np.expm1(pred_val_log)
        mae_val = mean_absolute_error(y_val_euros_real, pred_val_euros)

        logger.info("[%d/%d] %s -> Validation MAE: %.2f €", idx, len(combinations), params, mae_val)
        if mae_val < best_mae:
            best_mae = mae_val
            best_params = params

    logger.info("Phase 1 Winner: %s (MAE: %.2f €)", best_params, best_mae)

    # Phase 2: Convergence Curve
    logger.info("Phase 2: Analyzing error convergence over ensemble size...")
    tree_range = list(range(25, 351, 25))
    mae_curve = []

    for n_trees in tree_range:
        rf = RandomForestRegressor(
            n_estimators=n_trees,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params,
        )
        rf.fit(X_train_trans, y_train_log)

        pred_val_log = np.clip(rf.predict(X_val_trans), -np.inf, 30)
        pred_val_euros = np.expm1(pred_val_log)
        val_mae = mean_absolute_error(y_val_euros_real, pred_val_euros)
        mae_curve.append(val_mae)
        logger.info("n_estimators=%d -> Val MAE: %.2f €", n_trees, val_mae)

    # Plot convergence curve
    plot_convergence_curve(
        tree_range,
        mae_curve,
        "Random Forest Error Convergence: Breiman's Empirical Limit",
        out_dir / "rf_amount_convergence_curve.png",
        ylabel="Validation MAE (€)",
        color="#2CA02C",
    )

    pd.DataFrame({"n_estimators": tree_range, "mae_val": mae_curve}).to_csv(out_dir / "convergence_data.csv", index=False)
    logger.info("Tuning complete. Optimum parameters: %s with n_estimators=%d", best_params, tree_range[int(np.argmin(mae_curve))])


if __name__ == "__main__":
    run_tuning_rf_amount()
