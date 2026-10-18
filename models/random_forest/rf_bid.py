"""Final Random Forest Regressor for tender bid discount prediction.

Implements the winning configuration found during hyperparameter tuning for percentage discount:
- Features preprocessing with ColumnTransformer.
- Directly models percentage discount [0, 100]%.
- Inverts predictions to euros using tender budget for apples-to-apples evaluation.
- Overfitting diagnosis (Train vs Test).
- Aggregated feature importance with one-hot parent mapping.
- Error diagnostic plots and SHAP explanation.
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
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from config import (
    BUDGET_COL,
    EXTRA_LEAKAGE_BID,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    RANDOM_STATE,
    RF_PARAMS_BID,
    SHAP_SAMPLE_SIZE,
    TARGET_BID_PCT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    get_dataset_path,
)
from src.data_cleaning import clean_dataframe, drop_leakage_columns
from src.feature_importance import aggregate_importances
from src.metrics import evaluate_bid_in_euros
from src.preprocessing import build_preprocessor
from src.visualization import (
    plot_boxplot_ape,
    plot_feature_importance,
    plot_scatter_real_vs_pred,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_rf_bid() -> None:
    """Train and evaluate winning Random Forest model for tender bid discount."""
    out_dir = OUTPUT_DIR / "random_forest_bid"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_df = pd.read_parquet(get_dataset_path("train"))
        val_df = pd.read_parquet(get_dataset_path("val"))
        test_df = pd.read_parquet(get_dataset_path("test"))
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    # Clean datasets and calculate TARGET_BID_PCT
    train_df = clean_dataframe(train_df, target="bid_pct")
    val_df = clean_dataframe(val_df, target="bid_pct")
    test_df = clean_dataframe(test_df, target="bid_pct")

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    y_train_pct = full_train_df[TARGET_BID_PCT].copy()
    y_test_pct = test_df[TARGET_BID_PCT].values
    budget_test = test_df[BUDGET_COL].values
    y_test_euros_real = budget_test * (1.0 - y_test_pct / 100.0)

    # Drop leakage columns
    drop_cols = LEAKAGE_COLS + EXTRA_LEAKAGE_BID
    full_train_df = drop_leakage_columns(full_train_df, drop_cols)
    test_df = drop_leakage_columns(test_df, drop_cols)

    X_train = full_train_df.drop(columns=[TARGET_BID_PCT], errors="ignore")
    X_test = test_df.drop(columns=[TARGET_BID_PCT], errors="ignore")

    # 1. Baseline: 0% discount
    metrics_baseline = evaluate_bid_in_euros(
        "Trivial Baseline (0% discount)",
        y_test_euros_real,
        np.zeros_like(y_test_pct),
        budget_test,
    )

    # 2. Build Pipeline
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = build_preprocessor(X_train)

    rf_model = RandomForestRegressor(
        **RF_PARAMS_BID,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline_rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf_model),
    ])

    logger.info("Training winning Random Forest (%% bid) with params: %s...", RF_PARAMS_BID)
    pipeline_rf.fit(X_train, y_train_pct)

    # Predict and evaluate in real euros
    y_pred_test_pct = pipeline_rf.predict(X_test)
    metrics_rf = evaluate_bid_in_euros("Random Forest Winner (% Bid -> €)", y_test_euros_real, y_pred_test_pct, budget_test)

    # 3. Overfitting Check
    y_pred_train_pct = pipeline_rf.predict(X_train)
    budget_train = full_train_df[BUDGET_COL].values
    y_train_euros_real = budget_train * (1.0 - y_train_pct.values / 100.0)
    y_pred_train_euros = budget_train * (1.0 - np.clip(y_pred_train_pct, 0.0, 100.0) / 100.0)

    train_r2 = r2_score(y_train_euros_real, y_pred_train_euros)
    train_mae = mean_absolute_error(y_train_euros_real, y_pred_train_euros)
    logger.info("Overfitting check -> Train R²: %.4f | Test R²: %.4f", train_r2, metrics_rf["R²"])
    logger.info("Overfitting check -> Train MAE: %.2f € | Test MAE: %.2f €", train_mae, metrics_rf["MAE (€)"])

    # 4. Results Table
    df_results = pd.DataFrame([metrics_baseline, metrics_rf])
    logger.info("Final Performance Comparison (In Euros):\n%s", df_results.to_markdown(index=False, floatfmt=",.4f"))
    df_results.to_csv(out_dir / "rf_bid_results.csv", index=False)

    # 5. Feature Importance
    try:
        df_imp = aggregate_importances(pipeline_rf, cat_cols=cat_cols, top_n=15)
        logger.info("Top 15 Most Important Features:\n%s", df_imp.to_markdown(index=False, floatfmt=",.4f"))
        plot_feature_importance(
            df_imp,
            "Feature Importance (Random Forest - % Bid Discount)",
            out_dir / "feature_importance_rf_bid.png",
        )
        df_imp.to_csv(out_dir / "feature_importance_rf_bid.csv", index=False)
    except Exception as err:
        logger.warning("Feature importance plotting failed: %s", err)

    # 6. Performance Plots
    y_pred_rf_euros = budget_test * (1.0 - np.clip(y_pred_test_pct, 0.0, 100.0) / 100.0)
    df_plot = pd.DataFrame({
        "APE Baseline": 100 * np.abs(y_test_euros_real - budget_test) / y_test_euros_real,
        "APE RF": 100 * np.abs(y_test_euros_real - y_pred_rf_euros) / y_test_euros_real,
    })
    plot_boxplot_ape(
        df_plot,
        ["APE Baseline", "APE RF"],
        "APE Distribution: Baseline vs Random Forest (% Bid Discount)",
        out_dir / "boxplot_ape_rf_bid.png",
        palette=["#4C72B0", "#E1812C"],
    )

    plot_scatter_real_vs_pred(
        y_test_euros_real,
        y_pred_rf_euros,
        "Random Forest (% Bid Formulation): Real vs Predicted Award (Euros)",
        out_dir / "scatter_rf_bid_euros.png",
        color="#E1812C",
    )

    # 7. SHAP Values
    try:
        logger.info("Calculating SHAP summary plot for RF Bid...")
        feature_names = pipeline_rf.named_steps["preprocessor"].get_feature_names_out()
        X_test_trans = pipeline_rf.named_steps["preprocessor"].transform(X_test)
        X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

        sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_df))
        X_sample = X_test_df.sample(sample_size, random_state=RANDOM_STATE)

        explainer = shap.TreeExplainer(pipeline_rf.named_steps["model"])
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=20, show=False)
        plt.title("Feature Impact on % Bid Prediction (SHAP - Random Forest)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary_rf_bid.png", bbox_inches="tight", dpi=300)
        plt.close()
        logger.info("SHAP plot saved successfully.")
    except Exception as err:
        logger.warning("SHAP calculation skipped or failed: %s", err)

    # 8. Save Fitted Model Pipeline
    model_file = out_dir / "rf_bid_winner.joblib"
    joblib.dump(pipeline_rf, model_file)
    logger.info("Model pipeline saved to: %s", model_file)


if __name__ == "__main__":
    run_rf_bid()
