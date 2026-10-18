"""Unit tests for data cleaning pipelines and data leakage suppression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import BUDGET_COL, ESTIMATED_VALUE_COL, TARGET_AMOUNT, TARGET_BID_PCT
from src.data_cleaning import clean_dataframe, drop_leakage_columns


@pytest.fixture
def raw_test_df() -> pd.DataFrame:
    """Create a sample synthetic raw dataframe with edge-cases."""
    return pd.DataFrame({
        "es_exito": [1, 0, 1, 1, 1],
        TARGET_AMOUNT: [1000.0, 5000.0, 15000.0, 0.50, 8000.0],
        BUDGET_COL: [1200.0, 5000.0, 10000.0, 100.0, 10000.0],
        ESTIMATED_VALUE_COL: [1200.0, 5000.0, 12000.0, 100.0, 10000.0],
        "_id": ["a", "b", "c", "d", "e"],
        "lote_numero_ofertas_recibidas": [3, 2, 1, 5, 4],
    })


def test_clean_dataframe_amount_filters(raw_test_df: pd.DataFrame) -> None:
    # Row 0: es_exito=1, 1000 <= 1200, >= 2.0 -> KEEP
    # Row 1: es_exito=0 -> FILTERED OUT
    # Row 2: 15000 > 10000 (Overcost) -> FILTERED OUT
    # Row 3: 0.50 < 2.0 (Symbolic price) -> FILTERED OUT
    # Row 4: es_exito=1, 8000 <= 10000, >= 2.0 -> KEEP
    cleaned = clean_dataframe(raw_test_df, target="amount")
    assert len(cleaned) == 2
    assert all(cleaned["es_exito"] == 1)
    assert all(cleaned[TARGET_AMOUNT] >= 2.0)
    assert all(cleaned[TARGET_AMOUNT] <= cleaned[BUDGET_COL])


def test_clean_dataframe_bid_pct_computation(raw_test_df: pd.DataFrame) -> None:
    cleaned = clean_dataframe(raw_test_df, target="bid_pct")
    assert TARGET_BID_PCT in cleaned.columns
    assert all(cleaned[TARGET_BID_PCT] >= 0.0)
    assert all(cleaned[TARGET_BID_PCT] <= 100.0)


def test_drop_leakage_columns() -> None:
    df = pd.DataFrame({
        "feature_1": [1, 2],
        "leakage_col": [9, 8],
        "other": [5, 6],
    })
    cleaned = drop_leakage_columns(df, ["leakage_col", "non_existent_col"])
    assert "leakage_col" not in cleaned.columns
    assert "feature_1" in cleaned.columns
    assert "other" in cleaned.columns
