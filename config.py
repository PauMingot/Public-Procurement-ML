"""
Centralized configuration for the Public Procurement ML project.

All file paths, column names, data-leakage columns, and winning
hyperparameters live here so every script imports from a single
source of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Override with the DATA_DIR environment variable to run on a different machine.
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))

TRAIN_PATH = DATA_DIR / "train_procesado_v2_limpio.parquet"
VAL_PATH = DATA_DIR / "val_procesado_v2_limpio.parquet"
TEST_PATH = DATA_DIR / "test_procesado_v2_limpio.parquet"

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Column names (matching the original PLACE dataset schema)
# ---------------------------------------------------------------------------
TARGET_AMOUNT = "lote_importe_adjudicacion_sin_impuestos"
TARGET_BID_PCT = "pct_baja"
BUDGET_COL = "lote_presupuesto_base_sin_impuestos"
ESTIMATED_VALUE_COL = "valor_estimado_imputado"

# ---------------------------------------------------------------------------
# Columns to drop (data-leakage prevention)
# ---------------------------------------------------------------------------
LEAKAGE_COLS = [
    "_id",
    "id",
    "fecha_primera_publicacion",
    "objeto",
    "lote_objeto",
    "organo_contratacion",
    "lote_adjudicatario",
    "lote_resultado",
    "cpv_final_imputado",
    "cif_normalizado",
    "lote_importe_adjudicacion_con_impuestos",
    "es_exito",
    "es_sobrecoste",
    "lote_numero_ofertas_recibidas",
    "presupuesto_medio_hist",
    "descuento_promedio",
    "lote_precio_oferta_mas_alta",
    "lote_precio_oferta_mas_baja",
    "presupuesto_base_sin_impuestos",
]

# Extra leakage columns only relevant for certain targets
EXTRA_LEAKAGE_AMOUNT = ["duracion_proceso_dias", "es_anomalia_temporal"]
EXTRA_LEAKAGE_BID = [TARGET_AMOUNT, "duracion_proceso_dias", "es_anomalia_temporal"]

# ---------------------------------------------------------------------------
# Winning hyperparameters (found during tuning)
# ---------------------------------------------------------------------------
RF_PARAMS_AMOUNT = {
    "n_estimators": 300,
    "max_depth": 15,
    "max_features": 0.8,
    "min_samples_leaf": 2,
}

RF_PARAMS_BID = {
    "n_estimators": 300,
    "max_depth": 20,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
}

XGB_PARAMS_AMOUNT = {
    "n_estimators": 209,
    "learning_rate": 0.2,
    "max_depth": 4,
    "reg_lambda": 1,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

XGB_PARAMS_BID = {
    "n_estimators": 235,
    "learning_rate": 0.05,
    "max_depth": 15,
    "reg_lambda": 40,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
MIN_AWARD_PRICE = 2.0  # euros – filter out symbolic prices
SHAP_SAMPLE_SIZE = 2000
