"""
Feature importance aggregation for tree-based pipelines.

One-hot encoding creates many dummy columns per categorical variable.
This module re-aggregates their importances back to the original
feature name so bar-charts are interpretable.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


def _map_to_original(col_name: str, cat_cols: list[str]) -> str:
    """Map a transformed column name back to its original feature."""
    if col_name.startswith("num__"):
        return col_name.replace("num__", "")
    if col_name.startswith("cat__"):
        for cat_col in cat_cols:
            if col_name.startswith(f"cat__{cat_col}_"):
                return cat_col
    return col_name


def aggregate_importances(
    pipeline: Pipeline,
    cat_cols: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract and aggregate feature importances from a fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline whose last step exposes
        ``feature_importances_``.
    cat_cols : list[str]
        Original categorical column names (before one-hot encoding).
    top_n : int, default 15
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Columns ``["feature", "importance"]`` sorted descending, with
        one-hot columns re-aggregated to their parent feature.
    """
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    df_raw = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_raw["original_feature"] = df_raw["feature"].apply(
        lambda x: _map_to_original(x, cat_cols)
    )

    df_agg = (
        df_raw.groupby("original_feature")["importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    return df_agg.head(top_n)
