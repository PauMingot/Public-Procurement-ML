"""Exploratory Data Analysis (EDA) for bidding companies and market dynamics.

Analyzes contractor behavior, market concentration, geographical dispersion,
degree of specialization, and historical discount aggressiveness.
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
)
from src.data_cleaning import clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_eda_companies() -> None:
    """Run company-level behavior and market structure analysis."""
    eda_output_dir = OUTPUT_DIR / "eda_companies"
    eda_output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    logger.info("Loading datasets for company analysis from %s, %s, %s...", TRAIN_PATH.name, VAL_PATH.name, TEST_PATH.name)
    try:
        full_df = pd.concat([
            pd.read_parquet(TRAIN_PATH),
            pd.read_parquet(VAL_PATH),
            pd.read_parquet(TEST_PATH),
        ], ignore_index=True)
    except FileNotFoundError as err:
        logger.error("Dataset not found: %s. Please ensure parquet files exist in data directory.", err)
        return

    logger.info("Total initial records: %d", len(full_df))
    full_df = clean_dataframe(full_df, target="amount")
    logger.info("Records after business filters: %d", len(full_df))

    # Extract latest snapshot per company
    company_col = "lote_adjudicatario"
    if company_col not in full_df.columns:
        logger.warning("Column '%s' not found. Skipping company profile extraction.", company_col)
        return

    if "fecha_primera_publicacion" in full_df.columns:
        full_df["fecha_primera_publicacion"] = pd.to_datetime(full_df["fecha_primera_publicacion"])
        full_df = full_df.sort_values("fecha_primera_publicacion")

    df_profiles = full_df.drop_duplicates(subset=[company_col], keep="last").copy()
    logger.info("Unique bidding companies detected: %d", len(df_profiles))

    # Clean history artifacts
    if "descuento_medio_hist" in df_profiles.columns:
        df_clean = df_profiles[df_profiles["descuento_medio_hist"] >= 0.0].copy()
    else:
        df_clean = df_profiles.copy()

    # 1. Distribution of Average Historical Discounts
    logger.info("1. Generating Historical Discounts distribution...")
    if "descuento_medio_hist" in df_clean.columns:
        plt.figure(figsize=(10, 6))
        data_desc = df_clean["descuento_medio_hist"] * 100
        sns.histplot(data_desc, bins=40, kde=True, color="teal")
        plt.axvline(data_desc.median(), color="red", linestyle="--", linewidth=2, label=f"Median: {data_desc.median():.2f}%")
        plt.axvline(data_desc.mean(), color="orange", linestyle="-", linewidth=2, label=f"Mean: {data_desc.mean():.2f}%")
        plt.title("Distribution of Contractor Historical Average Discount", fontsize=14, fontweight="bold")
        plt.xlabel("Historical Average Discount (%)", fontsize=12)
        plt.ylabel("Number of Companies", fontsize=12)
        plt.legend(title="Statistics", frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(eda_output_dir / "1_contractor_discount_distribution.png", dpi=300)
        plt.close()

    # 2. Market Concentration (Log-Log plot of wins)
    logger.info("2. Generating Market Concentration plot...")
    if "n_licitaciones_hist" in df_clean.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_clean["n_licitaciones_hist"], log_scale=True, color="orange")
        plt.title("Market Concentration: Number of Historical Wins per Company", fontsize=14, fontweight="bold")
        plt.xlabel("Historical Wins (Logarithmic Scale)", fontsize=12)
        plt.ylabel("Number of Companies", fontsize=12)
        for tick in [1, 10, 100, 1000]:
            plt.axvline(tick, color="grey", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(eda_output_dir / "2_market_concentration_log.png", dpi=300)
        plt.close()

    # 3. Scatter: Average Budget vs Average Discount
    logger.info("3. Generating Contractor Size vs Discount scatter...")
    if "presupuesto_medio_hist" in df_clean.columns and "descuento_medio_hist" in df_clean.columns:
        plt.figure(figsize=(10, 6))
        df_scatter = df_clean[df_clean["presupuesto_medio_hist"] > 100]
        sns.scatterplot(
            data=df_scatter,
            x="presupuesto_medio_hist",
            y="descuento_medio_hist",
            alpha=0.2,
            s=20,
            color="purple",
        )
        plt.xscale("log")
        plt.title("Contractor Average Project Size vs Average Discount", fontsize=14, fontweight="bold")
        plt.xlabel("Average Base Budget (€) [Log Scale]", fontsize=12)
        plt.ylabel("Average Discount Rate", fontsize=12)
        plt.tight_layout()
        plt.savefig(eda_output_dir / "3_contractor_size_vs_discount.png", dpi=300)
        plt.close()

    # 4. Advanced: Geographic scope & Specialization
    cols_geo = [c for c in df_clean.columns if "pct_hist_com_aut_" in c]
    cols_tipo = [c for c in df_clean.columns if "pct_hist_" in c and "com_aut" not in c]

    if cols_geo:
        df_clean["num_ccaa_operativas"] = (df_clean[cols_geo] > 0).sum(axis=1)
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_clean, x="num_ccaa_operativas", color="royalblue")
        plt.title("Geographical Scope of Bidders: Local vs National", fontsize=14, fontweight="bold")
        plt.xlabel("Number of Autonomous Communities with Won Contracts", fontsize=12)
        plt.ylabel("Number of Companies (Log Scale)", fontsize=12)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(eda_output_dir / "4_geographic_scope.png", dpi=300)
        plt.close()

    if cols_tipo:
        df_clean["grado_especializacion"] = df_clean[cols_tipo].max(axis=1) * 100
        plt.figure(figsize=(10, 6))
        sns.histplot(df_clean["grado_especializacion"], bins=20, color="green", kde=True)
        plt.title("Contractor Specialization Degree", fontsize=14, fontweight="bold")
        plt.xlabel("Maximum Share in Single Contract Type (%)", fontsize=12)
        plt.ylabel("Number of Companies", fontsize=12)
        plt.axvline(df_clean["grado_especializacion"].median(), color="red", linestyle="--", label="Median")
        plt.legend()
        plt.tight_layout()
        plt.savefig(eda_output_dir / "5_specialization_distribution.png", dpi=300)
        plt.close()

    # 5. Experience vs Aggressiveness (Learning curve)
    if "n_licitaciones_hist" in df_clean.columns and "descuento_medio_hist" in df_clean.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_clean,
            x="n_licitaciones_hist",
            y="descuento_medio_hist",
            alpha=0.3,
            color="darkred",
            s=30,
            label="Bidding Companies",
        )
        try:
            sns.regplot(
                data=df_clean,
                x="n_licitaciones_hist",
                y="descuento_medio_hist",
                scatter=False,
                lowess=True,
                color="blue",
                line_kws={"linestyle": "--", "linewidth": 2},
            )
        except Exception:
            pass

        plt.xscale("log")
        plt.title("Learning Curve: Contractor Experience vs Bidding Aggressiveness", fontsize=14, fontweight="bold")
        plt.xlabel("Historical Won Tenders (Log Scale)", fontsize=12)
        plt.ylabel("Average Discount Rate", fontsize=12)
        plt.legend(loc="upper right", frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(eda_output_dir / "6_experience_vs_discount_curve.png", dpi=300)
        plt.close()

    logger.info("Company EDA completed successfully. Figures saved to: %s", eda_output_dir)


if __name__ == "__main__":
    run_eda_companies()
