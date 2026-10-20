"""Metrics and diagnostic plots for binary classifiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PALETTE = ["#1f5f8b", "#d1495b", "#edae49", "#00798c", "#6a4c93", "#30638e"]


def classification_metrics(y_true, y_score, threshold: float = 0.5, prefix: str = "") -> dict:
    """Threshold-dependent and threshold-free metrics in one flat dict."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }
    return {f"{prefix}{k}": (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def _finish(fig: plt.Figure, path: Path | None):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_overlay(curves: dict[str, tuple], y_true, path: Path | None = None) -> plt.Figure:
    """ROC curves of several models on the same axes. ``curves[name] = y_score``."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for color, (name, score) in zip(PALETTE, curves.items(), strict=False):
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(
            fpr, tpr, color=color, lw=2, label=f"{name} (AUC {roc_auc_score(y_true, score):.3f})"
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="chance")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curves (hold-out test)",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_pr_overlay(curves: dict[str, tuple], y_true, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    prevalence = float(np.mean(y_true))
    for color, (name, score) in zip(PALETTE, curves.items(), strict=False):
        p, r, _ = precision_recall_curve(y_true, score)
        ax.plot(
            r,
            p,
            color=color,
            lw=2,
            label=f"{name} (AP {average_precision_score(y_true, score):.3f})",
        )
    ax.axhline(
        prevalence, color="k", ls="--", lw=1, alpha=0.6, label=f"prevalence {prevalence:.2f}"
    )
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (hold-out test)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_calibration(curves: dict[str, tuple], y_true, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for color, (name, score) in zip(PALETTE, curves.items(), strict=False):
        frac, mean_pred = calibration_curve(y_true, score, n_bins=6, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", color=color, lw=1.5, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfectly calibrated")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed fraction positive",
        title="Reliability diagram",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_confusion(
    y_true, y_score, threshold: float, title: str, path: Path | None = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["no disease", "disease"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    return _finish(fig, path)


def permutation_table(model, X, y, seed: int, n_repeats: int = 15) -> pd.DataFrame:
    """Permutation importance on the *raw* input columns (pipeline-aware)."""
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_importances(table: pd.DataFrame, title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    t = table.iloc[::-1]
    ax.barh(
        t["feature"], t["importance_mean"], xerr=t["importance_std"], color=PALETTE[0], alpha=0.85
    )
    ax.set(xlabel="Mean drop in ROC-AUC when shuffled", title=title)
    ax.grid(axis="x", alpha=0.3)
    return _finish(fig, path)


def plot_leaderboard(board: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """CV ROC-AUC with error bars next to the hold-out ROC-AUC for every model family."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(board))
    w = 0.38
    ax.bar(
        x - w / 2,
        board["cv_roc_auc_mean"],
        w,
        yerr=board["cv_roc_auc_std"],
        capsize=4,
        color=PALETTE[0],
        label="CV ROC-AUC (mean ± std)",
    )
    ax.bar(x + w / 2, board["test_roc_auc"], w, color=PALETTE[2], label="Hold-out ROC-AUC")
    ax.set_xticks(x, board["model"], rotation=15)
    lo = max(0.5, float(min(board["cv_roc_auc_mean"].min(), board["test_roc_auc"].min())) - 0.08)
    ax.set_ylim(lo, 1.0)
    ax.set(ylabel="ROC-AUC", title="Model leaderboard")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, path)
