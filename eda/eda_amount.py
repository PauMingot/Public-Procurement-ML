"""Exploratory Data Analysis (EDA) for tender award amounts.

Generates distributions, budget vs award scatter plots, correlation matrices,
and categorical breakdowns for public procurement contracts in Spain.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path for modular execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    BUDGET_COL,
    ESTIMATED_VALUE_COL,
    OUTPUT_DIR,
    TARGET_AMOUNT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    get_dataset_path,
)
from src.data_cleaning import clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_eda_amount() -> None:
    """Run full EDA pipeline for award amount."""
    eda_output_dir = OUTPUT_DIR / "eda_amount"
    eda_output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    try:
        train_path = get_dataset_path("train")
        if train_path == get_dataset_path("sample"):
            df_eda = pd.read_parquet(train_path)
        else:
            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(get_dataset_path("val"))
            test_df = pd.read_parquet(get_dataset_path("test"))
            df_eda = pd.concat([train_df, val_df, test_df], ignore_index=True)
    except FileNotFoundError as err:
        logger.error("Dataset not found: %s. Please ensure parquet files exist in data directory.", err)
        return
    logger.info("Total initial records: %d", len(df_eda))

    # Apply standardized business cleaning
    df_eda = clean_dataframe(df_eda, target="amount")
    logger.info("Records after business filters: %d", len(df_eda))

    # 1. Distribution of Target (Log scale)
    logger.info("1. Generating Award Amount distribution...")
    plt.figure(figsize=(10, 6))
    mean_val = df_eda[TARGET_AMOUNT].mean()
    median_val = df_eda[TARGET_AMOUNT].median()

    sns.histplot(df_eda[TARGET_AMOUNT], bins=80, kde=True, log_scale=True, color="#4C72B0", alpha=0.7)
    plt.axvline(mean_val, color="#C44E52", linestyle="--", linewidth=2, label=f"Mean: {mean_val:,.0f} €")
    plt.axvline(median_val, color="#55A868", linestyle="-", linewidth=2, label=f"Median: {median_val:,.0f} €")
    plt.title("Distribution of Tender Award Amount (Log Scale)", fontsize=14, fontweight="bold")
    plt.xlabel("Amount (€) - Log Scale")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(eda_output_dir / "1_amount_distribution_log.png", dpi=300)
    plt.close()

    # 2. Scatter: Budget vs Award (Zoom P99)
    logger.info("2. Generating Budget vs Award scatter plot...")
    if BUDGET_COL in df_eda.columns:
        plt.figure(figsize=(8, 8))
        q99_budget = float(df_eda[BUDGET_COL].quantile(0.99))
        q99_amount = float(df_eda[TARGET_AMOUNT].quantile(0.99))
        max_plot = max(q99_budget, q99_amount)

        sns.scatterplot(data=df_eda, x=BUDGET_COL, y=TARGET_AMOUNT, alpha=0.15, color="#4C72B0", s=12)
        plt.plot([1, max_plot], [1, max_plot], color="#C44E52", linestyle="--", linewidth=2, label="y = x (No discount)")
        plt.xlim(max(1.0, float(df_eda[BUDGET_COL].quantile(0.01))), max_plot)
        plt.ylim(max(1.0, float(df_eda[TARGET_AMOUNT].quantile(0.01))), max_plot)
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Base Budget vs Award Amount\n(Zoom: excluding top 1% extremes)", fontsize=13, fontweight="bold")
        plt.xlabel("Base Budget (€) - Log Scale")
        plt.ylabel("Award Amount (€) - Log Scale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(eda_output_dir / "2_scatter_budget_vs_award_zoom.png", dpi=300)
        plt.close()

    # 3. Correlation matrices
    logger.info("3. Generating correlation matrices...")
    cols_corr = [
        TARGET_AMOUNT,
        BUDGET_COL,
        ESTIMATED_VALUE_COL,
        "peso_relativo_lote",
        "es_loteado",
        "lote",
    ]
    present_cols = [c for c in cols_corr if c in df_eda.columns]
    if len(present_cols) > 1:
        rename_map = {
            TARGET_AMOUNT: "Award Amount",
            BUDGET_COL: "Base Budget",
            ESTIMATED_VALUE_COL: "Estimated Value",
            "peso_relativo_lote": "Lot Relative Weight",
            "es_loteado": "Is Multi-Lot",
            "lote": "Lot Number",
        }
        df_corr = df_eda[present_cols].rename(columns=rename_map)

        # Pearson
        plt.figure(figsize=(8, 7))
        sns.heatmap(df_corr.corr(method="pearson"), annot=True, cmap="vlag", fmt=".3f", vmin=-1, vmax=1, square=True)
        plt.title("Pearson Correlation Matrix (Linear Relationships)", fontsize=13, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(eda_output_dir / "3a_correlation_pearson.png", dpi=300)
        plt.close()

        # Spearman
        plt.figure(figsize=(8, 7))
        sns.heatmap(df_corr.corr(method="spearman"), annot=True, cmap="vlag", fmt=".3f", vmin=-1, vmax=1, square=True)
        plt.title("Spearman Correlation Matrix (Monotonic Relationships)", fontsize=13, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(eda_output_dir / "3b_correlation_spearman.png", dpi=300)
        plt.close()

    # 4. Categorical boxplots
    logger.info("4. Generating categorical boxplots...")
    cat_vars = ["tipo_contrato", "organo_cluster_label", "tipo_procedimiento", "sistema_contratacion"]
    for var in cat_vars:
        if var in df_eda.columns:
            plt.figure(figsize=(12, 7))
            order = df_eda.groupby(var)[TARGET_AMOUNT].median().sort_values(ascending=False).index
            sns.boxplot(
                data=df_eda,
                x=TARGET_AMOUNT,
                y=var,
                order=order,
                palette="Set2",
                showfliers=True,
                fliersize=2,
                flierprops={"marker": "o", "alpha": 0.3, "color": "gray"},
            )
            plt.xscale("log")
            plt.title(f"Award Amount by {var.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
            plt.xlabel("Award Amount (€) - Log Scale")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(eda_output_dir / f"4_boxplot_{var}.png", dpi=300)
            plt.close()

    logger.info("EDA completed successfully. Figures saved to: %s", eda_output_dir)


if __name__ == "__main__":
    run_eda_amount()
