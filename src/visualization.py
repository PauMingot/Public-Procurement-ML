"""
Reusable plotting functions for model evaluation and EDA.

All figures are saved to disk and ``plt.close()``'d to avoid memory
leaks in long-running scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_feature_importance(
    df_importance: pd.DataFrame,
    title: str,
    output_path: Path,
    palette: str = "viridis",
) -> None:
    """Horizontal bar chart of aggregated feature importances.

    Parameters
    ----------
    df_importance : pd.DataFrame
        Must contain columns ``"original_feature"`` and ``"importance"``.
    title : str
        Plot title.
    output_path : Path
        File path to save the figure.
    palette : str, default "viridis"
        Seaborn colour palette name.
    """
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df_importance,
        x="importance",
        y="original_feature",
        hue="original_feature",
        legend=False,
        palette=palette,
    )
    plt.title(title)
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_scatter_real_vs_pred(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
    *,
    log_scale: bool = True,
    color: str = "#E1812C",
) -> None:
    """Scatter plot of real vs. predicted values with the y=x diagonal.

    Parameters
    ----------
    y_real : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.
    title : str
        Plot title.
    output_path : Path
        File path to save the figure.
    log_scale : bool, default True
        Whether to use log-log axes.
    color : str, default "#E1812C"
        Scatter dot colour.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_real, y_pred, alpha=0.15, s=10, color=color)

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
        qmin = max(10.0, float(np.quantile(y_real, 0.005)))
        qmax = float(np.quantile(y_real, 0.995))
        plt.xlim(qmin, qmax)
        plt.ylim(qmin, qmax)
        plt.plot(
            [qmin, qmax], [qmin, qmax],
            color="red", linestyle="--", linewidth=2,
            label="Perfect prediction (y = x)",
        )
    else:
        lo = min(y_real.min(), y_pred.min()) - 5
        hi = max(y_real.max(), y_pred.max()) + 5
        plt.plot(
            [lo, hi], [lo, hi],
            color="black", linestyle="--", linewidth=2,
            label="Perfect prediction (y = x)",
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Actual Value", fontsize=12)
    plt.ylabel("Predicted Value", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_boxplot_ape(
    df_results: pd.DataFrame,
    ape_columns: list[str],
    title: str,
    output_path: Path,
    palette: list[str] | None = None,
    median_lines: dict[str, str] | None = None,
) -> None:
    """Horizontal box-plot comparing APE distributions across models.

    Parameters
    ----------
    df_results : pd.DataFrame
        Must contain the columns listed in *ape_columns*.
    ape_columns : list[str]
        Column names with Absolute Percentage Error values.
    title : str
        Plot title.
    output_path : Path
        File path to save the figure.
    palette : list[str] | None
        Colours for each box.
    median_lines : dict[str, str] | None
        ``{column_name: colour}`` for vertical median lines.
    """
    if palette is None:
        palette = ["#4C72B0", "#E1812C"]

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df_results[ape_columns],
        orient="h",
        showfliers=False,
        palette=palette,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Absolute Percentage Error (%)")
    plt.gca().set_xscale("log")
    plt.xlim(0.1, 300)

    if median_lines:
        for col, colour in median_lines.items():
            med = df_results[col].median()
            plt.axvline(
                x=med, color=colour, linestyle="--",
                label=f"MdAPE {col} ({med:.2f}%)",
            )
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_convergence_curve(
    n_trees: list[int],
    scores: list[float],
    title: str,
    output_path: Path,
    *,
    ylabel: str = "MAE (€) – Validation Set",
    color: str = "#E1812C",
) -> None:
    """Plot MAE vs. number of trees for ensemble convergence analysis.

    Parameters
    ----------
    n_trees : list[int]
        Number of trees tested.
    scores : list[float]
        Corresponding MAE scores.
    title : str
        Plot title.
    output_path : Path
        File path to save the figure.
    ylabel : str
        Y-axis label.
    color : str
        Line colour.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, scores, marker="o", linestyle="-", color=color, linewidth=2)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Number of Trees (K)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    best_idx = int(np.argmin(scores))
    best_n = n_trees[best_idx]
    plt.axvline(
        x=best_n, color="red", linestyle="--",
        label=f"Optimum (K = {best_n})",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
