"""Linear regression models (Ridge, Lasso, ElasticNet) for tender award amount prediction.

Implements:
1. Baseline (Award = Budget) and simple univariate log-log regression.
2. Multivariate penalized regressions (Ridge, Lasso, ElasticNet) with hyperparameter search.
3. Feature coefficient shrinkage interpretation (Lasso sparsity).
4. Euro-scale evaluation (R², MAE, MdAPE) and error diagnostic plots.
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
    TARGET_AMOUNT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.data_cleaning import clean_dataframe
from src.metrics import evaluate_in_euros, mdape
from src.visualization import plot_boxplot_ape, plot_scatter_real_vs_pred

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_regression_amount() -> None:
    """Train and evaluate linear models for award amount."""
    out_dir = OUTPUT_DIR / "linear_amount"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets from %s...", TRAIN_PATH.parent)
    try:
        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        test_df = pd.read_parquet(TEST_PATH)
    except FileNotFoundError as err:
        logger.error("Dataset missing: %s", err)
        return

    train_df = clean_dataframe(train_df, target="amount")
    val_df = clean_dataframe(val_df, target="amount")
    test_df = clean_dataframe(test_df, target="amount")

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # PredefinedSplit: -1 for train, 0 for validation
    test_fold = np.concatenate([
        np.full(len(train_df), -1),
        np.zeros(len(val_df)),
    ])
    ps = PredefinedSplit(test_fold)

    y_train_log = np.log1p(full_train_df[TARGET_AMOUNT])
    y_test_euros_real = test_df[TARGET_AMOUNT].values
    presupuesto_test = test_df[BUDGET_COL].values

    # 1. Baseline: Award = Budget
    metrics_baseline = {
        "Model": "Trivial Baseline (Budget = Award)",
        "R²": r2_score(y_test_euros_real, presupuesto_test),
        "MAE (€)": mean_absolute_error(y_test_euros_real, presupuesto_test),
        "MdAPE (%)": mdape(y_test_euros_real, presupuesto_test),
    }

    # 2. Simple log-log regression (Univariate: Budget -> Award)
    X_train_simple = np.log1p(full_train_df[BUDGET_COL]).values.reshape(-1, 1)
    X_test_simple = np.log1p(test_df[BUDGET_COL]).values.reshape(-1, 1)
    model_simple = LinearRegression().fit(X_train_simple, y_train_log)
    metrics_simple = evaluate_in_euros("Simple Linear Regression (Log Budget)", y_test_euros_real, model_simple.predict(X_test_simple))

    # 3. Multivariate Regularized Regressions
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

    # Log-transform numeric features
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

    # Pipeline configurations
    pipe_ridge = Pipeline([("preprocessor", preprocessor), ("model", Ridge(random_state=RANDOM_STATE))])
    pipe_lasso = Pipeline([("preprocessor", preprocessor), ("model", Lasso(random_state=RANDOM_STATE, max_iter=10000))])
    pipe_elastic = Pipeline([("preprocessor", preprocessor), ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=10000))])

    logger.info("Optimizing Ridge regression...")
    grid_ridge = GridSearchCV(pipe_ridge, {"model__alpha": [0.1, 1.0, 10.0, 100.0, 500.0]}, cv=ps, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train_m, y_train_log)
    best_ridge = grid_ridge.best_estimator_

    logger.info("Optimizing Lasso regression...")
    grid_lasso = GridSearchCV(pipe_lasso, {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}, cv=ps, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train_m, y_train_log)
    best_lasso = grid_lasso.best_estimator_

    logger.info("Optimizing ElasticNet regression...")
    grid_elastic = GridSearchCV(pipe_elastic, {"model__alpha": [0.0001, 0.001, 0.01, 0.1], "model__l1_ratio": [0.2, 0.5, 0.8]}, cv=ps, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train_m, y_train_log)
    best_elastic = grid_elastic.best_estimator_

    metrics_ridge = evaluate_in_euros("Ridge Regression (Tuned)", y_test_euros_real, best_ridge.predict(X_test_m))
    metrics_lasso = evaluate_in_euros("Lasso Regression (Tuned)", y_test_euros_real, best_lasso.predict(X_test_m))
    metrics_elastic = evaluate_in_euros("ElasticNet Regression (Tuned)", y_test_euros_real, best_elastic.predict(X_test_m))

    # Overfitting Check on Lasso
    y_train_euros_real = full_train_df[TARGET_AMOUNT].values
    y_pred_train_lasso = np.expm1(np.clip(best_lasso.predict(X_train_m), -np.inf, 30))
    train_mae_lasso = mean_absolute_error(y_train_euros_real, y_pred_train_lasso)
    logger.info("Lasso Overfitting Check -> Train MAE: %.2f € | Test MAE: %.2f €", train_mae_lasso, metrics_lasso["MAE (€)"])

    # Comparative table
    df_comparison = pd.DataFrame([metrics_baseline, metrics_simple, metrics_ridge, metrics_lasso, metrics_elastic])
    logger.info("Linear Models Summary:\n%s", df_comparison.to_markdown(index=False, floatfmt=",.4f"))
    df_comparison.to_csv(out_dir / "linear_amount_comparison.csv", index=False)

    # Lasso coefficient interpretation
    try:
        feature_names = best_lasso.named_steps["preprocessor"].get_feature_names_out()
        coefs = best_lasso.named_steps["model"].coef_
        df_coef = pd.DataFrame({"Variable": feature_names, "Coefficient": coefs})
        df_coef["AbsCoef"] = df_coef["Coefficient"].abs()
        df_coef = df_coef.sort_values("AbsCoef", ascending=False).head(15)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_coef, x="Coefficient", y="Variable", palette="vlag")
        plt.title("Top Feature Coefficients (Lasso Regression in Log Space)", fontsize=13, fontweight="bold")
        plt.xlabel("Log-scale Coefficient Impact")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(out_dir / "lasso_feature_coefficients.png", dpi=300)
        plt.close()
    except Exception as err:
        logger.warning("Could not generate coefficient plot: %s", err)

    # Diagnostic plots
    y_pred_lasso_euros = np.expm1(np.clip(best_lasso.predict(X_test_m), -np.inf, 30))
    plot_scatter_real_vs_pred(
        y_test_euros_real,
        y_pred_lasso_euros,
        "Lasso Regression: Real vs Predicted Award Amount",
        out_dir / "scatter_lasso_amount.png",
        color="#55A868",
    )

    df_ape = pd.DataFrame({
        "APE Baseline": 100 * np.abs(y_test_euros_real - presupuesto_test) / y_test_euros_real,
        "APE Lasso": 100 * np.abs(y_test_euros_real - y_pred_lasso_euros) / y_test_euros_real,
    })
    plot_boxplot_ape(
        df_ape,
        ["APE Baseline", "APE Lasso"],
        "APE Distribution: Baseline vs Lasso Regression",
        out_dir / "boxplot_ape_lasso.png",
        palette=["#4C72B0", "#55A868"],
    )

    # Save model artifact
    joblib.dump(best_lasso, out_dir / "lasso_amount_model.joblib")
    logger.info("Best model saved to: %s", out_dir / "lasso_amount_model.joblib")


if __name__ == "__main__":
    run_regression_amount()
