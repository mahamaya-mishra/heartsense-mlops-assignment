"""Exploratory data analysis: figures to reports/figures/ and a JSON profile to reports/.

python -m heartsense.explore
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heartsense import settings as cfg
from heartsense.ingest import load_clean, summary

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
CLASS_COLORS = {0: "#1f5f8b", 1: "#d1495b"}
CLASS_NAMES = {0: "no disease", 1: "disease"}


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_class_balance(df: pd.DataFrame, out: Path) -> Path:
    counts = df[cfg.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(
        [CLASS_NAMES[i] for i in counts.index],
        counts.values,
        color=[CLASS_COLORS[i] for i in counts.index],
    )
    for b, n in zip(bars, counts.values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, n + 2, f"{n} ({n / len(df):.0%})", ha="center")
    ax.set(title="Class balance", ylabel="patients")
    return _save(fig, "01_class_balance", out)


def fig_missingness(df: pd.DataFrame, out: Path) -> Path:
    miss = df.isna().sum()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(miss.index, miss.values, color="#edae49")
    ax.set(title=f"Missing values per column (total {int(miss.sum())})", ylabel="count")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "02_missing_values", out)


def fig_continuous_distributions(df: pd.DataFrame, out: Path) -> Path:
    cols = cfg.CONTINUOUS_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3.4))
    for ax, col in zip(axes, cols, strict=True):
        for cls in (0, 1):
            sns.kdeplot(
                df.loc[df[cfg.TARGET] == cls, col].dropna().astype(float),
                ax=ax,
                fill=True,
                alpha=0.35,
                color=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls],
                warn_singular=False,
            )
        ax.set(title=col, ylabel="")
    axes[0].legend(fontsize=8)
    fig.suptitle("Continuous features by outcome", y=1.03)
    return _save(fig, "03_continuous_by_outcome", out)


def fig_categorical_prevalence(df: pd.DataFrame, out: Path) -> Path:
    cols = cfg.NOMINAL_FEATURES + cfg.BINARY_FEATURES + cfg.COUNT_FEATURES
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.3 * ((n + 1) // 2), 6.5))
    for ax, col in zip(axes.ravel(), cols, strict=False):
        grp = df.groupby(col, observed=True)[cfg.TARGET].agg(["mean", "size"]).reset_index()
        labels = [cfg.CODEBOOK[col].get(int(v), str(v)) for v in grp[col]]
        bars = ax.bar(labels, grp["mean"], color="#00798c")
        for b, s in zip(bars, grp["size"], strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"n={s}",
                ha="center",
                fontsize=7,
            )
        ax.axhline(df[cfg.TARGET].mean(), ls="--", color="k", lw=1)
        ax.set(title=col, ylim=(0, 1.05), ylabel="disease rate")
        ax.tick_params(axis="x", rotation=25, labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Disease prevalence by category (dashed line = overall rate)", y=1.01)
    return _save(fig, "04_prevalence_by_category", out)


def fig_correlation(df: pd.DataFrame, out: Path) -> Path:
    corr = df.astype(float).corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot_kws={"size": 7},
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Spearman correlation (lower triangle)")
    return _save(fig, "05_correlation_heatmap", out)


def fig_age_hr_scatter(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for cls in (0, 1):
        sub = df[df[cfg.TARGET] == cls]
        ax.scatter(
            sub["age"].astype(float),
            sub["max_heart_rate"].astype(float),
            s=28,
            alpha=0.65,
            color=CLASS_COLORS[cls],
            label=CLASS_NAMES[cls],
            edgecolor="white",
            linewidth=0.4,
        )
    ages = np.linspace(28, 78, 50)
    ax.plot(ages, 220 - ages, "k--", lw=1, label="220 - age (predicted max)")
    ax.set(
        xlabel="age (years)",
        ylabel="max heart rate achieved",
        title="Chronotropic response by outcome",
    )
    ax.legend(fontsize=8)
    return _save(fig, "06_age_vs_max_heart_rate", out)


def fig_st_depression_box(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    tmp = df.dropna(subset=["st_slope"]).copy()
    tmp["slope"] = tmp["st_slope"].map(lambda v: cfg.CODEBOOK["st_slope"][int(v)])
    tmp["outcome"] = tmp[cfg.TARGET].map(CLASS_NAMES)
    sns.boxplot(
        data=tmp,
        x="slope",
        y="st_depression",
        hue="outcome",
        palette=list(CLASS_COLORS.values()),
        ax=ax,
    )
    ax.set(title="ST depression by ST slope and outcome", xlabel="peak exercise ST slope")
    return _save(fig, "07_st_depression_by_slope", out)


def profile(df: pd.DataFrame) -> dict:
    """Numbers quoted in the report and asserted by the tests."""
    numeric = df[cfg.CONTINUOUS_FEATURES].astype(float).describe().round(2)
    corr_with_target = (
        df.astype(float)
        .corr(method="spearman")[cfg.TARGET]
        .drop(cfg.TARGET)
        .abs()
        .sort_values(ascending=False)
    )
    prevalence_by = {
        col: {
            cfg.CODEBOOK[col].get(int(k), str(k)): round(float(v), 3)
            for k, v in df.groupby(col, observed=True)[cfg.TARGET].mean().items()
        }
        for col in cfg.NOMINAL_FEATURES + cfg.BINARY_FEATURES
    }
    return {
        **summary(df),
        "continuous_summary": numeric.to_dict(),
        "top_correlates_with_target": corr_with_target.head(6).round(3).to_dict(),
        "prevalence_by_category": prevalence_by,
    }


def run(
    df: pd.DataFrame | None = None,
    figures_dir: Path = cfg.FIGURES_DIR / "eda",
    reports_dir: Path = cfg.REPORTS_DIR,
) -> dict:
    df = load_clean() if df is None else df
    paths = [
        fig_class_balance(df, figures_dir),
        fig_missingness(df, figures_dir),
        fig_continuous_distributions(df, figures_dir),
        fig_categorical_prevalence(df, figures_dir),
        fig_correlation(df, figures_dir),
        fig_age_hr_scatter(df, figures_dir),
        fig_st_depression_box(df, figures_dir),
    ]
    prof = profile(df)
    prof["figures"] = [
        str(p.relative_to(cfg.PROJECT_ROOT)) if p.is_relative_to(cfg.PROJECT_ROOT) else str(p)
        for p in paths
    ]
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "eda_profile.json").write_text(json.dumps(prof, indent=2))
    log.info("wrote %d figures to %s", len(paths), figures_dir)
    return prof


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    prof = run()
    print(
        json.dumps(
            {
                k: prof[k]
                for k in ("rows", "positive_rate", "missing_cells", "top_correlates_with_target")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
