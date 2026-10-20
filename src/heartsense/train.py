"""Train, tune, compare, track and register heart-disease classifiers.

Five model families are tuned with randomised search under repeated stratified k-fold
cross-validation. Every sampled configuration is a nested MLflow run; every family has a parent
run with CV + hold-out metrics, plots and the serialised pipeline. The champion is the family
with the best **mean CV ROC-AUC** (the hold-out set is never used for selection). It is
registered in the MLflow Model Registry under the alias ``production`` and exported to
``model_store/`` for the API image.

    python -m heartsense.train            # full search (~3-5 min on a laptop)
    python -m heartsense.train --quick    # tiny search for CI smoke runs and tests
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from heartsense import __version__
from heartsense import settings as cfg
from heartsense.evaluate import (
    classification_metrics,
    permutation_table,
    plot_calibration,
    plot_confusion,
    plot_importances,
    plot_leaderboard,
    plot_pr_overlay,
    plot_roc_overlay,
)
from heartsense.ingest import file_sha256, load_clean
from heartsense.pipeline import build_pipeline, output_feature_names, split_xy

log = logging.getLogger(__name__)

SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}
PRIMARY_METRIC = "roc_auc"


def search_space(quick: bool = False) -> dict[str, tuple[object, dict, int]]:
    """``family -> (estimator, param_distributions, n_iter)``."""
    n = 3 if quick else 24
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=cfg.SEED),
            {
                "classifier__C": loguniform(1e-3, 1e2),
                "classifier__class_weight": [None, "balanced"],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
            n,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=cfg.SEED, n_jobs=-1),
            {
                "classifier__n_estimators": randint(150, 600),
                "classifier__max_depth": [3, 4, 5, 6, 8, None],
                "classifier__min_samples_leaf": randint(1, 10),
                "classifier__max_features": ["sqrt", 0.4, 0.6],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
            n,
        ),
        "svm_rbf": (
            SVC(kernel="rbf", probability=True, random_state=cfg.SEED),
            {
                "classifier__C": loguniform(1e-2, 1e2),
                "classifier__gamma": loguniform(1e-3, 1e0),
                "classifier__class_weight": [None, "balanced"],
            },
            n,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=cfg.SEED, early_stopping=False),
            {
                "classifier__learning_rate": loguniform(1e-2, 3e-1),
                "classifier__max_iter": randint(60, 300),
                "classifier__max_leaf_nodes": randint(4, 24),
                "classifier__min_samples_leaf": randint(5, 30),
                "classifier__l2_regularization": uniform(0.0, 2.0),
            },
            n,
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": randint(5, 40),
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
            max(2, n // 2),
        ),
    }


# ------------------------------------------------------------------------------ helpers
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cfg.PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _jsonable(params: dict) -> dict:
    """Strip the ``classifier__`` prefix and make every value JSON/MLflow friendly."""
    out = {}
    for key, value in params.items():
        if hasattr(value, "item"):  # numpy scalar -> python scalar
            value = value.item()
        if not (isinstance(value, (int, float, str, bool)) or value is None):
            value = str(value)
        out[key.removeprefix("classifier__")] = value
    return out


def _cv_row(search: RandomizedSearchCV, idx: int) -> dict[str, float]:
    res = search.cv_results_
    row = {}
    for name in SCORING:
        row[f"cv_{name}_mean"] = float(res[f"mean_test_{name}"][idx])
        row[f"cv_{name}_std"] = float(res[f"std_test_{name}"][idx])
    row["cv_train_roc_auc_mean"] = float(res["mean_train_roc_auc"][idx])
    row["cv_fit_time_s"] = float(res["mean_fit_time"][idx])
    return row


def configure_mlflow(tracking_uri: str | None = None, experiment: str | None = None) -> str:
    """Point MLflow at the configured store and experiment (resolved at call time)."""
    mlflow.set_tracking_uri(tracking_uri or cfg.MLFLOW_TRACKING_URI)
    experiment = experiment or cfg.MLFLOW_EXPERIMENT
    exp = mlflow.get_experiment_by_name(experiment)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment)
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def _log_fig(fig: plt.Figure, artifact_path: str, local: Path | None = None) -> None:
    mlflow.log_figure(fig, artifact_path)
    if local is not None:
        local.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(local, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ one family
def tune_family(
    family: str,
    estimator,
    space: dict,
    n_iter: int,
    X_train,
    y_train,
    X_test,
    y_test,
    tags: dict,
    figures_dir: Path,
    quick: bool,
) -> dict:
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.CV_SPLITS, n_repeats=1 if quick else cfg.CV_REPEATS, random_state=cfg.SEED
    )
    search = RandomizedSearchCV(
        build_pipeline(estimator),
        space,
        n_iter=n_iter,
        scoring=SCORING,
        refit=PRIMARY_METRIC,
        cv=cv,
        n_jobs=-1,
        random_state=cfg.SEED,
        return_train_score=True,
        error_score="raise",
    )
    with mlflow.start_run(
        run_name=family, tags={**tags, "family": family, "stage": "family"}
    ) as parent:
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        # nested runs: one per sampled configuration
        ranks = search.cv_results_["rank_test_" + PRIMARY_METRIC]
        for i, params in enumerate(search.cv_results_["params"]):
            with mlflow.start_run(
                run_name=f"{family}/trial-{i:02d}",
                nested=True,
                tags={"family": family, "stage": "trial"},
            ):
                mlflow.log_params(_jsonable(params))
                mlflow.log_metrics({**_cv_row(search, i), "rank": float(ranks[i])})

        best = search.best_estimator_
        cv_metrics = _cv_row(search, search.best_index_)
        test_scores = best.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(
            y_test, test_scores, cfg.DECISION_THRESHOLD, prefix="test_"
        )

        mlflow.log_params(
            {
                **_jsonable(search.best_params_),
                "n_iter": n_iter,
                "cv_splits": cfg.CV_SPLITS,
                "cv_repeats": cv.get_n_splits() // cfg.CV_SPLITS,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "threshold": cfg.DECISION_THRESHOLD,
            }
        )
        mlflow.log_metrics({**cv_metrics, **test_metrics, "search_seconds": elapsed})

        _log_fig(
            plot_confusion(
                y_test, test_scores, cfg.DECISION_THRESHOLD, f"{family}: hold-out confusion"
            ),
            "plots/confusion_matrix.png",
            figures_dir / f"{family}_confusion.png",
        )
        imp = permutation_table(best, X_test, y_test, cfg.SEED, n_repeats=5 if quick else 15)
        mlflow.log_table(imp, "tables/permutation_importance.json")
        _log_fig(
            plot_importances(imp, f"{family}: permutation importance (test)"),
            "plots/permutation_importance.png",
            figures_dir / f"{family}_importance.png",
        )
        mlflow.log_dict(
            {"transformed_features": output_feature_names(best)},
            "features/transformed_columns.json",
        )
        mlflow.log_table(
            pd.DataFrame(search.cv_results_).drop(columns=["params"]), "tables/cv_results.json"
        )

        signature = infer_signature(X_train.head(), best.predict_proba(X_train.head()))
        info = mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=X_train.head(3).astype("float64"),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        log.info(
            "%-24s cv_roc_auc=%.3f±%.3f test_roc_auc=%.3f test_recall=%.3f (%.0fs)",
            family,
            cv_metrics["cv_roc_auc_mean"],
            cv_metrics["cv_roc_auc_std"],
            test_metrics["test_roc_auc"],
            test_metrics["test_recall"],
            elapsed,
        )
        return {
            "model": family,
            "run_id": parent.info.run_id,
            "model_uri": info.model_uri,
            "estimator": best,
            "test_scores": test_scores,
            "best_params": _jsonable(search.best_params_),
            **cv_metrics,
            **test_metrics,
        }


# ----------------------------------------------------------------------------- driver
def run(
    quick: bool = False,
    data_path: Path = cfg.CLEAN_DATA_PATH,
    model_store: Path = cfg.MODEL_STORE,
    reports_dir: Path = cfg.REPORTS_DIR,
    register: bool = True,
) -> dict:
    figures_dir = Path(reports_dir) / "figures" / "model"
    df = load_clean(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_FRACTION, stratify=y, random_state=cfg.SEED
    )
    log.info("train=%d test=%d positive_rate=%.3f", len(X_train), len(X_test), y.mean())

    exp_id = configure_mlflow()
    tags = {
        "git_sha": _git_sha(),
        "heartsense_version": __version__,
        "dataset_sha256": cfg.DATASET_SHA256[:12],
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "mode": "quick" if quick else "full",
    }

    results = [
        tune_family(
            name, est, space, n_iter, X_train, y_train, X_test, y_test, tags, figures_dir, quick
        )
        for name, (est, space, n_iter) in search_space(quick).items()
    ]

    # champion = best mean CV ROC-AUC, ties broken by CV recall (we prefer not to miss disease)
    results.sort(key=lambda r: (round(r["cv_roc_auc_mean"], 4), r["cv_recall_mean"]), reverse=True)
    champion = results[0]
    log.info("champion: %s", champion["model"])

    board_cols = [
        "model",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "cv_train_roc_auc_mean",
        "cv_accuracy_mean",
        "cv_recall_mean",
        "cv_precision_mean",
        "cv_f1_mean",
        "test_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_f1",
        "test_average_precision",
        "test_brier",
        "test_false_negatives",
        "test_false_positives",
        "cv_fit_time_s",
    ]
    board = pd.DataFrame([{k: r[k] for k in board_cols} for r in results])
    reports_dir.mkdir(parents=True, exist_ok=True)
    board.to_csv(reports_dir / "leaderboard.csv", index=False)
    (reports_dir / "leaderboard.md").write_text(board.round(3).to_markdown(index=False) + "\n")

    curves = {r["model"]: r["test_scores"] for r in results}
    with mlflow.start_run(
        run_name="model-selection",
        tags={**tags, "stage": "selection", "champion": champion["model"]},
    ):
        mlflow.log_table(board, "tables/leaderboard.json")
        mlflow.log_metrics(
            {
                "champion_cv_roc_auc": champion["cv_roc_auc_mean"],
                "champion_test_roc_auc": champion["test_roc_auc"],
            }
        )
        mlflow.log_param("selection_rule", "max mean CV ROC-AUC, tie-break CV recall")
        _log_fig(plot_leaderboard(board), "plots/leaderboard.png", figures_dir / "leaderboard.png")
        _log_fig(
            plot_roc_overlay(curves, y_test), "plots/roc_curves.png", figures_dir / "roc_curves.png"
        )
        _log_fig(
            plot_pr_overlay(curves, y_test), "plots/pr_curves.png", figures_dir / "pr_curves.png"
        )
        _log_fig(
            plot_calibration(curves, y_test),
            "plots/calibration.png",
            figures_dir / "calibration.png",
        )

    # ------------------------------------------------------------------ export + registry
    model_store = Path(model_store)
    model_store.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion["estimator"], model_store / cfg.MODEL_FILE.name, compress=3)

    version = None
    if register:
        mv = mlflow.register_model(champion["model_uri"], cfg.MLFLOW_REGISTERED_MODEL)
        client = MlflowClient()
        client.set_registered_model_alias(
            cfg.MLFLOW_REGISTERED_MODEL, cfg.MLFLOW_PRODUCTION_ALIAS, mv.version
        )
        client.set_model_version_tag(
            cfg.MLFLOW_REGISTERED_MODEL, mv.version, "family", champion["model"]
        )
        client.update_model_version(
            cfg.MLFLOW_REGISTERED_MODEL,
            mv.version,
            description=f"{champion['model']} selected by CV ROC-AUC "
            f"{champion['cv_roc_auc_mean']:.3f}; git {tags['git_sha']}",
        )
        version = int(mv.version)

    metadata = {
        "model_name": cfg.MLFLOW_REGISTERED_MODEL,
        "model_version": f"{champion['model']}-v{version or 0}",
        "family": champion["model"],
        "registry_version": version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "heartsense_version": __version__,
        "git_sha": tags["git_sha"],
        "mlflow_run_id": champion["run_id"],
        "mlflow_experiment_id": exp_id,
        "dataset": {
            "source": cfg.DATASET_URL,
            "sha256": cfg.DATASET_SHA256,
            "rows": len(df),
            "clean_file_sha256": file_sha256(Path(data_path)),
        },
        "split": {
            "test_fraction": cfg.TEST_FRACTION,
            "seed": cfg.SEED,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_positive_rate": round(float(y_train.mean()), 4),
        },
        "decision_threshold": cfg.DECISION_THRESHOLD,
        "risk_tiers": cfg.RISK_TIERS,
        "input_features": cfg.INPUT_FEATURES,
        "hyperparameters": champion["best_params"],
        "metrics": {k: champion[k] for k in board_cols if k not in ("model",)},
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (model_store / cfg.METADATA_FILE.name).write_text(json.dumps(metadata, indent=2, default=float))
    log.info("exported %s and metadata", model_store / cfg.MODEL_FILE.name)
    return {"champion": champion["model"], "leaderboard": board, "metadata": metadata}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="tiny search space (CI / tests)")
    parser.add_argument("--no-register", action="store_true", help="skip the MLflow model registry")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    out = run(quick=args.quick, register=not args.no_register)
    print(
        "\n"
        + out["leaderboard"][
            [
                "model",
                "cv_roc_auc_mean",
                "cv_roc_auc_std",
                "test_roc_auc",
                "test_recall",
                "test_accuracy",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nchampion: {out['champion']}  ->  {cfg.MODEL_FILE}")


if __name__ == "__main__":
    main()
