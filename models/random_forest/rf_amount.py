"""Final Random Forest Regressor for tender award amount prediction.

Implements the winning configuration found during hyperparameter tuning:
- Full preprocessing pipeline (numeric passthrough + categorical One-Hot).
- Direct log-transformed target training with exponential destransformation.
- Euro-scale performance metrics (R², MAE, MdAPE).
- Overfitting check comparing Train vs. Test degradation.
- Aggregated feature importances (mapping One-Hot dummies back to parent variables).
- Error distribution boxplots and log-log scatter plots.
- SHAP tree-explainability summary.
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
    EXTRA_LEAKAGE_AMOUNT,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    RANDOM_STATE,
    RF_PARAMS_AMOUNT,
    SHAP_SAMPLE_SIZE,
    TARGET_AMOUNT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
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


def run_rf_amount() -> None:
    """Train and evaluate final Random Forest model for tender award amount."""
    out_dir = OUTPUT_DIR / "random_forest_amount"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_df = pd.read_parquet(get_dataset_path("train"))
        val_df = pd.read_parquet(get_dataset_path("val"))
        test_df = pd.read_parquet(get_dataset_path("test"))
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    # Clean datasets
    train_df = clean_dataframe(train_df, target="amount")
    val_df = clean_dataframe(val_df, target="amount")
    test_df = clean_dataframe(test_df, target="amount")

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Drop leakage columns
    drop_cols = LEAKAGE_COLS + EXTRA_LEAKAGE_AMOUNT
    full_train_df = drop_leakage_columns(full_train_df, drop_cols)
    test_df = drop_leakage_columns(test_df, drop_cols)

    # Features and targets
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

    # 2. Build Pipeline with winning parameters
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = build_preprocessor(X_train)

    rf_model = RandomForestRegressor(
        **RF_PARAMS_AMOUNT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline_rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf_model),
    ])

    logger.info("Training winning Random Forest model with params: %s...", RF_PARAMS_AMOUNT)
    pipeline_rf.fit(X_train, y_train_log)

    # Predict in log space and evaluate in euros
    y_pred_test_log = pipeline_rf.predict(X_test)
    metrics_rf = evaluate_in_euros("Random Forest Winner (Amount)", y_test_euros_real, y_pred_test_log)

    # 3. Overfitting Diagnosis (Train vs Test in Euros)
    y_pred_train_log = pipeline_rf.predict(X_train)
    y_pred_train_euros = np.expm1(np.clip(y_pred_train_log, -np.inf, 30))
    train_r2 = r2_score(y_train.values, y_pred_train_euros)
    train_mae = mean_absolute_error(y_train.values, y_pred_train_euros)

    logger.info("Overfitting check -> Train R²: %.4f | Test R²: %.4f", train_r2, metrics_rf["R²"])
    logger.info("Overfitting check -> Train MAE: %.2f € | Test MAE: %.2f €", train_mae, metrics_rf["MAE (€)"])

    # 4. Results Summary
    df_results = pd.DataFrame([metrics_baseline, metrics_rf])
    logger.info("Final Performance Comparison:\n%s", df_results.to_markdown(index=False, floatfmt=",.4f"))
    df_results.to_csv(out_dir / "rf_amount_results.csv", index=False)

    # 5. Feature Importance Aggregation
    try:
        df_imp = aggregate_importances(pipeline_rf, cat_cols=cat_cols, top_n=15)
        logger.info("Top 15 Most Important Features:\n%s", df_imp.to_markdown(index=False, floatfmt=",.4f"))
        plot_feature_importance(
            df_imp,
            "Feature Importance (Random Forest - Amount Prediction)",
            out_dir / "feature_importance_rf_amount.png",
        )
        df_imp.to_csv(out_dir / "feature_importance_rf_amount.csv", index=False)
    except Exception as err:
        logger.warning("Feature importance plotting failed: %s", err)

    # 6. Performance & Error Plots
    y_pred_test_euros = np.expm1(np.clip(y_pred_test_log, -np.inf, 30))
    df_plot = pd.DataFrame({
        "APE Baseline": 100 * np.abs(y_test_euros_real - presupuesto_test) / y_test_euros_real,
        "APE RF": 100 * np.abs(y_test_euros_real - y_pred_test_euros) / y_test_euros_real,
    })
    plot_boxplot_ape(
        df_plot,
        ["APE Baseline", "APE RF"],
        "APE Distribution: Baseline vs Random Forest (Award Amount)",
        out_dir / "boxplot_ape_rf_amount.png",
        palette=["#4C72B0", "#E1812C"],
    )

    plot_scatter_real_vs_pred(
        y_test_euros_real,
        y_pred_test_euros,
        "Random Forest: Real vs Predicted Award Amount (Log Scale)",
        out_dir / "scatter_rf_amount.png",
        color="#E1812C",
    )

    # 7. SHAP Values
    try:
        logger.info("Calculating SHAP summary plot (sample of up to %d records)...", SHAP_SAMPLE_SIZE)
        feature_names = pipeline_rf.named_steps["preprocessor"].get_feature_names_out()
        X_test_trans = pipeline_rf.named_steps["preprocessor"].transform(X_test)
        X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

        sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_df))
        X_sample = X_test_df.sample(sample_size, random_state=RANDOM_STATE)

        explainer = shap.TreeExplainer(pipeline_rf.named_steps["model"])
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=20, show=False)
        plt.title("Feature Impact on Prediction (SHAP Summary - Random Forest)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary_rf_amount.png", bbox_inches="tight", dpi=300)
        plt.close()
        logger.info("SHAP plot saved successfully.")
    except Exception as err:
        logger.warning("SHAP calculation skipped or failed: %s", err)

    # 8. Save Fitted Model Pipeline
    model_file = out_dir / "rf_amount_winner.joblib"
    joblib.dump(pipeline_rf, model_file)
    logger.info("Model pipeline saved to: %s", model_file)


if __name__ == "__main__":
    run_rf_amount()
