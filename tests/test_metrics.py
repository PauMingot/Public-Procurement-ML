"""Unit tests for evaluation metrics (MdAPE, euro-scale conversions)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import evaluate_bid_in_euros, evaluate_in_euros, mdape


def test_mdape_perfect_prediction() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    assert mdape(y_true, y_pred) == 0.0


def test_mdape_known_percentage() -> None:
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([90.0, 90.0, 90.0])  # 10% error
    assert pytest.approx(mdape(y_true, y_pred), rel=1e-3) == 10.0


def test_mdape_zero_array() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([10.0, 20.0])
    assert mdape(y_true, y_pred) == 0.0


def test_evaluate_in_euros_output_structure() -> None:
    y_true_euros = np.array([1000.0, 5000.0, 10000.0])
    y_pred_log = np.log1p(np.array([950.0, 5100.0, 9900.0]))

    res = evaluate_in_euros("TestModel", y_true_euros, y_pred_log)
    assert res["Model"] == "TestModel"
    assert "R²" in res
    assert "MAE (€)" in res
    assert "MdAPE (%)" in res
    assert res["MAE (€)"] > 0
    assert res["MdAPE (%)"] >= 0


def test_evaluate_bid_in_euros() -> None:
    budget = np.array([100000.0, 200000.0])
    y_pred_pct = np.array([10.0, 20.0])  # 10% discount -> 90k, 20% discount -> 160k
    y_true_euros = np.array([90000.0, 160000.0])  # Exact matches

    res = evaluate_bid_in_euros("BidModel", y_true_euros, y_pred_pct, budget)
    assert pytest.approx(res["MAE (€)"], abs=1e-3) == 0.0
    assert pytest.approx(res["MdAPE (%)"], abs=1e-3) == 0.0
    assert pytest.approx(res["R²"], abs=1e-3) == 1.0
