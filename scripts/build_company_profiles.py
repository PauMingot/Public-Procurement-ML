"""Build lightweight company profile database for bidding simulation.

Extracts the latest historical track record per contractor:
- Computes specialization shares across contract types and regions.
- Computes average historical discounts and tender size volume.
- Saves an ultra-lightweight parquet file for the Streamlit simulator app.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import (
    BUDGET_COL,
    DATA_DIR,
    TARGET_AMOUNT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.data_cleaning import clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_company_profiles(output_path: Path | None = None) -> None:
    """Extract and persist unique contractor profiles."""
    if output_path is None:
        output_path = DATA_DIR / "perfiles_empresas.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    company_col = "lote_adjudicatario"
    logger.info("Loading train, val, and test datasets from %s...", DATA_DIR)
    try:
        df_train = pd.read_parquet(TRAIN_PATH)
        df_val = pd.read_parquet(VAL_PATH)
        df_test = pd.read_parquet(TEST_PATH)
    except FileNotFoundError as err:
        logger.error("Source parquet datasets missing: %s", err)
        return

    logger.info("Concatenating and applying business cleaning filters...")
    full_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    full_df = clean_dataframe(full_df, target="amount")

    if "fecha_primera_publicacion" in full_df.columns:
        full_df["fecha_primera_publicacion"] = pd.to_datetime(full_df["fecha_primera_publicacion"])
        full_df = full_df.sort_values("fecha_primera_publicacion")

    logger.info("Extracting latest snapshot for each contractor...")
    recent_profiles = full_df.drop_duplicates(subset=[company_col], keep="last").set_index(company_col)

    useful_cols = [c for c in recent_profiles.columns if any(k in c for k in ("hist", "media", "medio", "pct_"))]
    for c in ["fecha_primera_publicacion", "com_aut_adjudicador", "presupuesto_medio_hist"]:
        if c in recent_profiles.columns and c not in useful_cols:
            useful_cols.append(c)

    df_profiles = recent_profiles[useful_cols].copy()
    if "fecha_primera_publicacion" in df_profiles.columns:
        df_profiles.rename(columns={"fecha_primera_publicacion": "fecha_ultima_licitacion_empresa"}, inplace=True)

    # Filter invalid historical discounts (anomalous discounts >= 90%)
    if "descuento_medio_hist" in df_profiles.columns:
        df_profiles = df_profiles[df_profiles["descuento_medio_hist"] < 0.90]

    df_profiles.to_parquet(output_path)
    logger.info("Successfully exported %d contractor profiles to: %s", len(df_profiles), output_path)


if __name__ == "__main__":
    build_company_profiles()
