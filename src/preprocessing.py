"""
Preprocessing utilities for sklearn pipelines.

Builds a ``ColumnTransformer`` that passes numeric features through and
one-hot-encodes categoricals – the standard preprocessing shared by
every tree-based and linear model in this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer from a training feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Training features (after dropping target and leakage columns).

    Returns
    -------
    ColumnTransformer
        Ready to ``.fit()`` / ``.fit_transform()`` on training data.
    """
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="passthrough",
    )
