"""
Data cleaning pipelines for Spanish public procurement records.

Implements the strict business-logic filters that guarantee data quality
before any modelling step:

- Keep only successfully awarded tenders (``es_exito == 1``).
- Remove over-costs (award ≤ budget).
- Filter incoherent estimated values.
- Discard symbolic prices (< 2 €) and missing targets.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from config import (
    BUDGET_COL,
    ESTIMATED_VALUE_COL,
    MIN_AWARD_PRICE,
    TARGET_AMOUNT,
    TARGET_BID_PCT,
)

logger = logging.getLogger(__name__)


def clean_dataframe(
    df: pd.DataFrame,
    target: Literal["amount", "bid_pct"],
) -> pd.DataFrame:
    """Apply the standard TFG cleaning filters.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe read from parquet.
    target : {"amount", "bid_pct"}
        Which target variable the pipeline is preparing for.

        * ``"amount"``  → filters on ``TARGET_AMOUNT`` (absolute euros).
        * ``"bid_pct"`` → computes ``pct_baja`` and bounds it to [0, 100].

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input dataframe.
    """
    start_len = len(df)
    df = df.copy()

    # --- If the target is bid percentage, compute it first ---------------
    if target == "bid_pct":
        if TARGET_AMOUNT in df.columns and BUDGET_COL in df.columns:
            df[TARGET_BID_PCT] = (
                (df[BUDGET_COL] - df[TARGET_AMOUNT]) / df[BUDGET_COL]
            ) * 100

    # --- Common filters --------------------------------------------------
    if "es_exito" in df.columns:
        df = df[df["es_exito"] == 1]

    if target == "amount":
        if TARGET_AMOUNT in df.columns and BUDGET_COL in df.columns:
            df = df[df[TARGET_AMOUNT] <= df[BUDGET_COL]]

    if BUDGET_COL in df.columns and ESTIMATED_VALUE_COL in df.columns:
        df = df[df[BUDGET_COL] <= df[ESTIMATED_VALUE_COL]]

    # --- Target-specific filters -----------------------------------------
    if target == "amount":
        if TARGET_AMOUNT in df.columns:
            df = df.dropna(subset=[TARGET_AMOUNT])
            df = df[df[TARGET_AMOUNT] > 0]
            df = df[df[TARGET_AMOUNT] >= MIN_AWARD_PRICE]

    elif target == "bid_pct":
        if TARGET_BID_PCT in df.columns:
            df = df.dropna(subset=[TARGET_BID_PCT])
            df = df[
                (df[TARGET_BID_PCT] >= 0.0) & (df[TARGET_BID_PCT] <= 100.0)
            ]

    # --- Economic coherence filter ---------------------------------------
    if BUDGET_COL in df.columns and ESTIMATED_VALUE_COL in df.columns:
        df = df[df[ESTIMATED_VALUE_COL] <= (df[BUDGET_COL] * 10)]

    logger.info("Cleaning: %d → %d rows.", start_len, len(df))
    return df


def drop_leakage_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Drop data-leakage columns, ignoring those not present.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Column names to remove.

    Returns
    -------
    pd.DataFrame
        Dataframe with specified columns removed.
    """
    to_drop = [c for c in columns if c in df.columns]
    return df.drop(columns=to_drop)
