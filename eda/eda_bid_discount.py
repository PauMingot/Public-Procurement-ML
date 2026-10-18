"""Exploratory Data Analysis (EDA) for tender bid discounts (percentage discount).

Analyzes the distribution of discount percentages (0-100%), correlations with
historical bidder behavior, and categorical variations across contract types.
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
    OUTPUT_DIR,
    TARGET_BID_PCT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.data_cleaning import clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_eda_bid_discount() -> None:
    """Run full EDA pipeline for tender bid discount percentage."""
    eda_output_dir = OUTPUT_DIR / "eda_bid"
    eda_output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    logger.info("Loading datasets for Bid Discount EDA from %s, %s, %s...", TRAIN_PATH.name, VAL_PATH.name, TEST_PATH.name)
    try:
        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        test_df = pd.read_parquet(TEST_PATH)
    except FileNotFoundError as err:
        logger.error("Dataset not found: %s. Please ensure parquet files exist in data directory.", err)
        return

    df_eda = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info("Total initial records: %d", len(df_eda))

    # Apply standardized business cleaning with bid_pct computation
    df_eda = clean_dataframe(df_eda, target="bid_pct")
    logger.info("Records after business filters: %d", len(df_eda))

    # 1. Distribution of Bid Discount (%)
    logger.info("1. Generating Bid Discount percentage distribution...")
    plt.figure(figsize=(10, 6))
    mean_val = float(df_eda[TARGET_BID_PCT].mean())
    median_val = float(df_eda[TARGET_BID_PCT].median())

    sns.histplot(df_eda[TARGET_BID_PCT], bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    plt.axvline(mean_val, color="#C44E52", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}%")
    plt.axvline(median_val, color="#55A868", linestyle="-", linewidth=2, label=f"Median: {median_val:.2f}%")
    plt.title("Distribution of Bid Discount Percentage (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Bid Discount (%)")
    plt.ylabel("Frequency (Number of Tenders)")
    plt.xlim(-2, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eda_output_dir / "1_bid_discount_distribution.png", dpi=300)
    plt.close()

    # 2. Correlation Matrix with historical and business metrics
    logger.info("2. Generating Spearman correlation matrix...")
    cols_corr = [
        TARGET_BID_PCT,
        "descuento_medio_hist",
        "duracion_proceso_dias",
        BUDGET_COL,
        "n_licitaciones_hist",
        "dias_desde_ultima_licitacion",
    ]
    present_cols = [c for c in cols_corr if c in df_eda.columns]

    if len(present_cols) > 1:
        rename_map = {
            TARGET_BID_PCT: "% Bid Discount (Target)",
            "descuento_medio_hist": "Historical Avg Discount",
            "duracion_proceso_dias": "Process Duration (Days)",
            BUDGET_COL: "Base Budget",
            "n_licitaciones_hist": "Historical Tenders Won",
            "dias_desde_ultima_licitacion": "Days Since Last Tender",
        }
        df_corr = df_eda[present_cols].rename(columns=rename_map)

        plt.figure(figsize=(9, 8))
        sns.heatmap(
            df_corr.corr(method="spearman"),
            annot=True,
            cmap="vlag",
            fmt=".3f",
            vmin=-1,
            vmax=1,
            square=True,
            annot_kws={"size": 11},
        )
        plt.title("Spearman Correlation (% Bid Discount vs Business Predictors)", fontsize=13, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(eda_output_dir / "2_correlation_spearman_bid.png", dpi=300)
        plt.close()

    # 3. Categorical boxplots
    logger.info("3. Generating categorical boxplots...")
    cat_vars = ["tipo_contrato", "tipo_procedimiento"]
    for var in cat_vars:
        if var in df_eda.columns:
            plt.figure(figsize=(12, 7))
            order = df_eda.groupby(var)[TARGET_BID_PCT].median().sort_values(ascending=False).index
            sns.boxplot(
                data=df_eda,
                x=TARGET_BID_PCT,
                y=var,
                order=order,
                palette="Set2",
                showfliers=False,
            )
            plt.title(f"Bid Discount (%) by {var.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
            plt.xlabel("Bid Discount (%)")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(eda_output_dir / f"3_boxplot_bid_{var}.png", dpi=300)
            plt.close()

    logger.info("Bid discount EDA completed successfully. Figures saved to: %s", eda_output_dir)


if __name__ == "__main__":
    run_eda_bid_discount()
