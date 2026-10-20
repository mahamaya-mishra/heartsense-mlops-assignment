"""Render docs/architecture.png (pure matplotlib, no graphviz dependency).

python docs/render_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("architecture.png")
BLUE, RED, GOLD, TEAL, GREY, PURPLE = (
    "#1f5f8b",
    "#d1495b",
    "#edae49",
    "#00798c",
    "#5c677d",
    "#6a4c93",
)


def box(ax, x, y, w, h, title, lines=(), color=BLUE, fs=8.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.4,
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.4,
            edgecolor=color,
            facecolor="none",
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.22,
        title,
        ha="center",
        va="top",
        fontsize=fs + 1.5,
        fontweight="bold",
        color=color,
    )
    for i, ln in enumerate(lines):
        ax.text(
            x + w / 2, y + h - 0.55 - i * 0.30, ln, ha="center", va="top", fontsize=fs, color="#222"
        )


def arrow(ax, p, q, color=GREY, label=None, rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=ls,
        )
    )
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(
            mx,
            my + 0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9},
        )


def lane(ax, y, h, label, color):
    ax.add_patch(
        FancyBboxPatch(
            (0.15, y),
            15.7,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            linewidth=1,
            edgecolor=color,
            facecolor=color,
            alpha=0.04,
            linestyle="--",
        )
    )
    ax.text(0.3, y + h - 0.15, label, fontsize=10, fontweight="bold", color=color, va="top")


fig, ax = plt.subplots(figsize=(16, 9.2))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.2)
ax.axis("off")
fig.suptitle("HeartSense: end-to-end MLOps architecture", fontsize=15, fontweight="bold", y=0.985)

# ------------------------------------------------------------------ build-time lane
lane(ax, 5.05, 3.95, "BUILD TIME  (developer laptop + GitHub Actions)", BLUE)
box(
    ax,
    0.4,
    5.6,
    2.3,
    2.6,
    "Data",
    [
        "UCI Heart Disease",
        "SHA-256 pinned download",
        "heartsense.ingest",
        "clean CSV committed",
        "heartsense.explore (EDA)",
    ],
    BLUE,
)
box(
    ax,
    3.1,
    5.6,
    2.6,
    2.6,
    "Training",
    [
        "sklearn Pipeline",
        "engineer -> impute ->",
        "encode/scale -> clf",
        "5 families x RandomizedSearch",
        "repeated 5-fold CV",
    ],
    BLUE,
)
box(
    ax,
    6.1,
    5.6,
    2.5,
    2.6,
    "MLflow",
    [
        "tracking: params, metrics,",
        "plots, tables, model",
        "nested trial runs",
        "Model Registry",
        "alias @production",
    ],
    PURPLE,
)
box(
    ax,
    9.0,
    5.6,
    3.1,
    2.6,
    "GitHub Actions",
    [
        "ruff lint + helm lint",
        "pytest (3.11, 3.12) + cov",
        "train & upload artifacts",
        "docker build + smoke test",
        "publish only if all green",
    ],
    TEAL,
)
box(
    ax,
    12.5,
    5.6,
    3.1,
    2.6,
    "Container registry",
    [
        "ghcr.io/<owner>/",
        "heartsense-api:<sha7>",
        ":latest",
        "multi-stage image",
        "non-root, healthcheck",
    ],
    GOLD,
)
arrow(ax, (2.7, 6.9), (3.1, 6.9), BLUE)
arrow(ax, (5.7, 6.9), (6.1, 6.9), BLUE, "log / register")
arrow(ax, (8.6, 6.9), (9.0, 6.9), BLUE, "model_store/")
arrow(ax, (12.1, 6.9), (12.5, 6.9), BLUE, "push")
ax.text(
    5.35,
    5.2,
    "git push  ->  pipeline: quality -> tests -> train -> container",
    fontsize=8.5,
    color=TEAL,
    style="italic",
)

# ------------------------------------------------------------------ run-time lane
lane(ax, 0.2, 4.6, "RUN TIME  (Kubernetes: minikube / any cluster, Helm chart)", RED)
box(
    ax,
    0.4,
    1.3,
    2.2,
    2.2,
    "Clients",
    ["curl / Swagger UI", "load generator", "JSON over HTTP"],
    GREY,
)
box(
    ax,
    3.1,
    1.3,
    2.4,
    2.2,
    "Service",
    ["type: LoadBalancer", ":8000 -> pods", "(Ingress optional)"],
    RED,
)
box(
    ax,
    6.0,
    0.6,
    4.0,
    3.6,
    "Deployment: heartsense-api",
    [
        "FastAPI + uvicorn (x2 replicas, HPA 2-5)",
        "POST /predict, /predict/batch",
        "GET /healthz /readyz /metadata /metrics",
        "pydantic validation (422)",
        "structured JSON logs (no PHI)",
        "startup/readiness/liveness probes",
        "PDB, resource limits, read-only FS",
    ],
    RED,
)
box(
    ax, 10.6, 2.5, 2.4, 1.7, "Prometheus", ["pod discovery via", "annotations", "alert rules"], GOLD
)
box(
    ax,
    13.3,
    2.5,
    2.4,
    1.7,
    "Grafana",
    ["provisioned dashboard", "traffic, errors, p95,", "prediction drift"],
    GOLD,
)
box(
    ax,
    10.6,
    0.5,
    5.1,
    1.6,
    "Logging",
    [
        "kubectl logs -> one JSON line per request",
        "request_id, path, status, latency, label, risk_tier",
    ],
    GREY,
)
arrow(ax, (2.6, 2.4), (3.1, 2.4), RED, "HTTP")
arrow(ax, (5.5, 2.4), (6.0, 2.4), RED)
arrow(ax, (10.0, 3.35), (10.6, 3.35), GOLD, "scrape /metrics")
arrow(ax, (13.0, 3.35), (13.3, 3.35), GOLD, "PromQL")
arrow(ax, (10.0, 1.3), (10.6, 1.3), GREY, "stdout")
arrow(ax, (14.0, 5.6), (8.5, 4.2), GOLD, "helm upgrade --install (image tag)", rad=-0.25, ls="--")

fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
