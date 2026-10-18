"""Hyperparameter tuning and convergence analysis for Random Forest (Bid Discount).

Phase 1: Architecture grid (max_depth, min_samples_leaf, max_features).
Phase 2: Ensemble size convergence curve over n_estimators evaluated in real euros.
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
from src.visualization import plot_convergence_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tuning_rf_bid() -> None:
    """Execute hyperparameter tuning for Random Forest on percentage bid discount."""
    out_dir = OUTPUT_DIR / "tuning_rf_bid"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training and validation sets for RF Bid tuning...")
    try:
        train_df = clean_dataframe(pd.read_parquet(TRAIN_PATH), target="bid_pct")
        val_df = clean_dataframe(pd.read_parquet(VAL_PATH), target="bid_pct")
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    # Real validation ground truth in euros for business evaluation
    budget_val = val_df[BUDGET_COL].values
    y_val_pct = val_df[TARGET_BID_PCT].values
    y_val_euros_real = budget_val * (1.0 - y_val_pct / 100.0)

    drop_cols = LEAKAGE_COLS + EXTRA_LEAKAGE_BID
    train_df = drop_leakage_columns(train_df, drop_cols)
    val_df = drop_leakage_columns(val_df, drop_cols)

    X_train = train_df.drop(columns=[TARGET_BID_PCT], errors="ignore")
    y_train = train_df[TARGET_BID_PCT].values

    X_val = val_df.drop(columns=[TARGET_BID_PCT], errors="ignore")

    logger.info("Transforming features...")
    preprocessor = build_preprocessor(X_train)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    # Phase 1: Architecture Grid
    logger.info("Phase 1: Architecture search...")
    param_grid = {
        "max_depth": [10, 15, 20],
        "min_samples_leaf": [2, 4, 8],
        "max_features": [0.5, 0.8, "sqrt"],
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logger.info("Testing %d parameter combinations...", len(combinations))

    best_mae = float("inf")
    best_params: dict = {}

    for idx, params in enumerate(combinations, start=1):
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        rf.fit(X_train_trans, y_train)

        pred_val_pct = np.clip(rf.predict(X_val_trans), 0.0, 100.0)
        pred_val_euros = budget_val * (1.0 - pred_val_pct / 100.0)
        mae_val_euros = mean_absolute_error(y_val_euros_real, pred_val_euros)

        logger.info("[%d/%d] %s -> Euro Validation MAE: %.2f €", idx, len(combinations), params, mae_val_euros)
        if mae_val_euros < best_mae:
            best_mae = mae_val_euros
            best_params = params

    logger.info("Phase 1 Winner: %s (MAE in Euros: %.2f €)", best_params, best_mae)

    # Phase 2: Convergence Curve
    logger.info("Phase 2: Analyzing ensemble size convergence...")
    tree_range = list(range(25, 351, 25))
    mae_curve = []

    for n_trees in tree_range:
        rf = RandomForestRegressor(
            n_estimators=n_trees,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params,
        )
        rf.fit(X_train_trans, y_train)

        pred_val_pct = np.clip(rf.predict(X_val_trans), 0.0, 100.0)
        pred_val_euros = budget_val * (1.0 - pred_val_pct / 100.0)
        val_mae = mean_absolute_error(y_val_euros_real, pred_val_euros)
        mae_curve.append(val_mae)
        logger.info("n_estimators=%d -> Val MAE: %.2f €", n_trees, val_mae)

    plot_convergence_curve(
        tree_range,
        mae_curve,
        "Random Forest Error Convergence: % Bid Formulation",
        out_dir / "rf_bid_convergence_curve.png",
        ylabel="Validation MAE (€)",
        color="#E1812C",
    )

    pd.DataFrame({"n_estimators": tree_range, "mae_val_euros": mae_curve}).to_csv(out_dir / "convergence_data.csv", index=False)
    logger.info("Tuning complete. Best configuration: %s with n_estimators=%d", best_params, tree_range[int(np.argmin(mae_curve))])


if __name__ == "__main__":
    run_tuning_rf_bid()
