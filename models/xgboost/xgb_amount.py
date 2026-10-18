"""Final XGBoost Regressor for tender award amount prediction.

Implements the winning configuration found during hyperparameter tuning:
- Gradient boosted decision trees on log-transformed targets.
- Bounded predictions preventing extrapolation beyond historical limits.
- Full performance metrics in real euros (R², MAE, MdAPE).
- Overfitting check comparing Train vs. Test degradation.
- Aggregated feature importance with one-hot parent grouping.
- Diagnostic visual analytics (APE boxplot, real vs predicted log-log scatter).
- SHAP TreeExplainer interpretation.
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from config import (
    BUDGET_COL,
    EXTRA_LEAKAGE_AMOUNT,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    RANDOM_STATE,
    SHAP_SAMPLE_SIZE,
    TARGET_AMOUNT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    XGB_PARAMS_AMOUNT,
    get_dataset_path,
)
from src.data_cleaning import clean_dataframe, drop_leakage_columns
from src.feature_importance import aggregate_importances
from src.metrics import evaluate_in_euros, mdape
from src.preprocessing import build_preprocessor
from src.visualization import (
    plot_boxplot_ape,
    plot_feature_importance,
    plot_scatter_real_vs_pred,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_xgb_amount() -> None:
    """Train and evaluate winning XGBoost model for tender award amount."""
    out_dir = OUTPUT_DIR / "xgboost_amount"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_df = pd.read_parquet(get_dataset_path("train"))
        val_df = pd.read_parquet(get_dataset_path("val"))
        test_df = pd.read_parquet(get_dataset_path("test"))
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    train_df = clean_dataframe(train_df, target="amount")
    val_df = clean_dataframe(val_df, target="amount")
    test_df = clean_dataframe(test_df, target="amount")

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    drop_cols = LEAKAGE_COLS + EXTRA_LEAKAGE_AMOUNT
    full_train_df = drop_leakage_columns(full_train_df, drop_cols)
    test_df = drop_leakage_columns(test_df, drop_cols)

    X_train = full_train_df.drop(columns=[TARGET_AMOUNT], errors="ignore")
    y_train = full_train_df[TARGET_AMOUNT].copy()
    y_train_log = np.log1p(y_train)

    X_test = test_df.drop(columns=[TARGET_AMOUNT], errors="ignore")
    y_test_euros_real = test_df[TARGET_AMOUNT].values
    presupuesto_test = test_df[BUDGET_COL].values

    # 1. Baseline: Award = Budget
    metrics_baseline = {
        "Model": "Trivial Baseline (Budget = Award)",
        "R²": r2_score(y_test_euros_real, presupuesto_test),
        "MAE (€)": mean_absolute_error(y_test_euros_real, presupuesto_test),
        "MdAPE (%)": mdape(y_test_euros_real, presupuesto_test),
    }

    # 2. Build Pipeline
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = build_preprocessor(X_train)

    xgb_model = XGBRegressor(
        **XGB_PARAMS_AMOUNT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mae",
    )

    pipeline_xgb = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb_model),
    ])

    logger.info("Training winning XGBoost amount model with parameters: %s...", XGB_PARAMS_AMOUNT)
    pipeline_xgb.fit(X_train, y_train_log)

    # 3. Predict & Evaluate in real euros
    max_train_log = float(y_train_log.max())
    y_pred_test_log = np.clip(pipeline_xgb.predict(X_test), 0, max_train_log)
    metrics_xgb = evaluate_in_euros("XGBoost Winner (Amount)", y_test_euros_real, y_pred_test_log)

    # 4. Overfitting Diagnosis
    y_pred_train_log = np.clip(pipeline_xgb.predict(X_train), 0, max_train_log)
    y_pred_train_euros = np.expm1(y_pred_train_log)
    train_r2 = r2_score(y_train.values, y_pred_train_euros)
    train_mae = mean_absolute_error(y_train.values, y_pred_train_euros)

    logger.info("Overfitting check -> Train R²: %.4f | Test R²: %.4f", train_r2, metrics_xgb["R²"])
    logger.info("Overfitting check -> Train MAE: %.2f € | Test MAE: %.2f €", train_mae, metrics_xgb["MAE (€)"])

    # 5. Results Summary
    df_results = pd.DataFrame([metrics_baseline, metrics_xgb])
    logger.info("Final Performance Comparison:\n%s", df_results.to_markdown(index=False, floatfmt=",.4f"))
    df_results.to_csv(out_dir / "xgb_amount_results.csv", index=False)

    # 6. Feature Importance
    try:
        df_imp = aggregate_importances(pipeline_xgb, cat_cols=cat_cols, top_n=15)
        logger.info("Top 15 Most Important Features:\n%s", df_imp.to_markdown(index=False, floatfmt=",.4f"))
        plot_feature_importance(
            df_imp,
            "Feature Importance (XGBoost - Amount Prediction)",
            out_dir / "feature_importance_xgb_amount.png",
        )
        df_imp.to_csv(out_dir / "feature_importance_xgb_amount.csv", index=False)
    except Exception as err:
        logger.warning("Feature importance plotting failed: %s", err)

    # 7. Diagnostic Plots
    y_pred_test_euros = np.expm1(y_pred_test_log)
    df_plot = pd.DataFrame({
        "APE Baseline": 100 * np.abs(y_test_euros_real - presupuesto_test) / y_test_euros_real,
        "APE XGB": 100 * np.abs(y_test_euros_real - y_pred_test_euros) / y_test_euros_real,
    })
    plot_boxplot_ape(
        df_plot,
        ["APE Baseline", "APE XGB"],
        "APE Distribution: Baseline vs XGBoost (Award Amount)",
        out_dir / "boxplot_ape_xgb_amount.png",
        palette=["#4C72B0", "#008080"],
    )

    plot_scatter_real_vs_pred(
        y_test_euros_real,
        y_pred_test_euros,
        "XGBoost: Real vs Predicted Award Amount (Log Scale)",
        out_dir / "scatter_xgb_amount.png",
        color="#008080",
    )

    # 8. SHAP TreeExplainer
    try:
        logger.info("Calculating SHAP summary plot for XGBoost Amount...")
        feature_names = pipeline_xgb.named_steps["preprocessor"].get_feature_names_out()
        X_test_trans = pipeline_xgb.named_steps["preprocessor"].transform(X_test)
        X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

        sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_df))
        X_sample = X_test_df.sample(sample_size, random_state=RANDOM_STATE)

        explainer = shap.TreeExplainer(pipeline_xgb.named_steps["model"])
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=20, show=False)
        plt.title("Feature Impact on Prediction (SHAP Summary - XGBoost)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary_xgb_amount.png", bbox_inches="tight", dpi=300)
        plt.close()
        logger.info("SHAP plot saved successfully.")
    except Exception as err:
        logger.warning("SHAP calculation skipped or failed: %s", err)

    # 9. Save Fitted Model Pipeline
    model_file = out_dir / "xgb_amount_winner.joblib"
    joblib.dump(pipeline_xgb, model_file)
    logger.info("Model pipeline saved to: %s", model_file)


if __name__ == "__main__":
    run_xgb_amount()
