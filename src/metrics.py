"""
Evaluation metrics for public procurement price prediction.

Provides domain-specific metrics:

- **MdAPE** (Median Absolute Percentage Error) – robust to outliers.
- **evaluate_in_euros** – converts log-space predictions back to euros
  and computes R², MAE, and MdAPE in a single call.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Median Absolute Percentage Error (%).

    Parameters
    ----------
    y_true : array-like
        Ground-truth values (must be > 0 for the metric to be defined).
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MdAPE expressed as a percentage.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    ape = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.median(ape) * 100)


def evaluate_in_euros(
    model_name: str,
    y_true_euros: np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float | str]:
    """Evaluate a log-space model in real euros.

    Applies ``expm1`` to invert the ``log1p`` transformation, clips
    extreme predictions, and returns a dictionary of standard metrics.

    Parameters
    ----------
    model_name : str
        Human-readable name for the results table.
    y_true_euros : array-like
        True award prices in euros.
    y_pred_log : array-like
        Predicted values in ``log1p`` space.

    Returns
    -------
    dict
        Keys: ``"Model"``, ``"R²"``, ``"MAE (€)"``, ``"MdAPE (%)"``.
    """
    y_pred_log_safe = np.clip(y_pred_log, -np.inf, 30)
    y_pred_euros = np.expm1(y_pred_log_safe)
    y_pred_euros = np.maximum(y_pred_euros, 0)

    return {
        "Model": model_name,
        "R²": r2_score(y_true_euros, y_pred_euros),
        "MAE (€)": mean_absolute_error(y_true_euros, y_pred_euros),
        "MdAPE (%)": mdape(y_true_euros, y_pred_euros),
    }


def evaluate_bid_in_euros(
    model_name: str,
    y_true_euros: np.ndarray,
    y_pred_bid_pct: np.ndarray,
    budget: np.ndarray,
) -> dict[str, float | str]:
    """Evaluate a bid-percentage model in real euros.

    Converts predicted bid discounts back to euro amounts using:
    ``predicted_euros = budget × (1 − pct_baja / 100)``.

    Parameters
    ----------
    model_name : str
        Human-readable name for the results table.
    y_true_euros : array-like
        True award prices in euros.
    y_pred_bid_pct : array-like
        Predicted bid discounts in percent (0–100).
    budget : array-like
        Tender budgets (without taxes) in euros.

    Returns
    -------
    dict
        Keys: ``"Model"``, ``"R²"``, ``"MAE (€)"``, ``"MdAPE (%)"``.
    """
    pct_safe = np.clip(y_pred_bid_pct, 0.0, 100.0)
    y_pred_euros = np.asarray(budget) * (1.0 - pct_safe / 100.0)

    return {
        "Model": model_name,
        "R²": r2_score(y_true_euros, y_pred_euros),
        "MAE (€)": mean_absolute_error(y_true_euros, y_pred_euros),
        "MdAPE (%)": mdape(y_true_euros, y_pred_euros),
    }
