"""Unit tests for Scikit-Learn preprocessing pipeline builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing import build_preprocessor


def test_build_preprocessor_fit_transform() -> None:
    X_train = pd.DataFrame({
        "num_col1": [1.0, 2.0, 3.0, 4.0],
        "num_col2": [10.0, 20.0, 30.0, 40.0],
        "cat_col": ["Services", "Works", "Supplies", "Services"],
    })

    preprocessor = build_preprocessor(X_train)
    assert isinstance(preprocessor, ColumnTransformer)

    transformed = preprocessor.fit_transform(X_train)
    # 2 numeric + 3 one-hot categories = 5 columns
    assert transformed.shape == (4, 5)


def test_preprocessor_handle_unknown_category() -> None:
    X_train = pd.DataFrame({
        "num_col": [1.0, 2.0],
        "cat_col": ["A", "B"],
    })
    X_test = pd.DataFrame({
        "num_col": [3.0],
        "cat_col": ["C"],  # Unknown category
    })

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    # Should not throw ValueError on unknown category
    transformed_test = preprocessor.transform(X_test)
    assert transformed_test.shape == (1, 3)
    # Unknown category should result in all zeros for one-hot
    assert transformed_test[0, 1] == 0.0
    assert transformed_test[0, 2] == 0.0
