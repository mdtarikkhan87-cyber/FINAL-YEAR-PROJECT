"""MLflow setup, artifact persistence, and plotting helpers.

Tracking is local and file-based: runs land in ``mlruns/`` at the project root,
which is gitignored. Nothing is sent anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a terminal session
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import DEFAULT_MLARTIFACT_DIR, default_tracking_uri  # noqa: E402


def setup_mlflow(experiment: str, tracking_uri: str | None = None,
                 artifact_dir: Path = DEFAULT_MLARTIFACT_DIR) -> str:
    """Point MLflow at the local SQLite store and select the experiment.

    MLflow 3.x refuses the old ``./mlruns`` file store outright, so tracking runs
    on SQLite. The artifact root is set explicitly at experiment-creation time --
    with a database backend MLflow would otherwise resolve artifacts relative to
    the current working directory, scattering them wherever the script was run
    from.
    """
    uri = tracking_uri or default_tracking_uri()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(experiment)
    return uri


def log_dataframe(df: pd.DataFrame, name: str, index: bool = False) -> None:
    """Log a DataFrame as a CSV artifact."""
    mlflow.log_text(df.to_csv(index=index), name)


def log_json(payload: dict, name: str) -> None:
    mlflow.log_text(json.dumps(payload, indent=2, default=str), name)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def confusion_figure(cm: pd.DataFrame, labels: list[str], title: str):
    """Row-normalised confusion matrix heatmap."""
    values = cm.to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    normed = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)

    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * len(labels), 1.4 + 1.0 * len(labels)))
    im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(values[i, j])}\n{normed[i, j]:.0%}",
                    ha="center", va="center", fontsize=8,
                    color="white" if normed[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row share")
    fig.tight_layout()
    return fig


def importance_figure(importance: pd.DataFrame, title: str, top_n: int = 20):
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(top) + 1.4))
    ax.barh(top["feature"], top["gain"], color="#3b6ea5")
    ax.set_xlabel("gain")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def residual_figure(y_true, y_pred, title: str):
    """Predicted-vs-actual and residual distribution, side by side."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[0].scatter(y_true, y_pred, s=5, alpha=0.18, color="#3b6ea5", edgecolors="none")
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect")
    axes[0].set_xlabel("actual days")
    axes[0].set_ylabel("predicted days")
    axes[0].set_title("predicted vs actual")
    axes[0].legend(loc="upper left", fontsize=8)

    resid = y_pred - y_true
    axes[1].hist(resid, bins=60, color="#3b6ea5")
    axes[1].axvline(0, color="r", ls="--", lw=1)
    axes[1].set_xlabel("residual (predicted - actual), days")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"residuals (mean {resid.mean():+.0f}, sd {resid.std():.0f})")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def log_and_save_figure(fig, mlflow_name: str, save_path: Path | None) -> None:
    mlflow.log_figure(fig, mlflow_name)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Local artifacts
# ---------------------------------------------------------------------------
def save_model_bundle(model, bundle: dict, artifact_dir: Path, stem: str) -> dict[str, Path]:
    """Persist the model twice: native XGBoost JSON plus a joblib bundle.

    The native file is the portable one (loadable by any XGBoost binding); the
    bundle carries the fitted estimator together with the category levels and
    feature order needed to reproduce a prediction, which the raw booster alone
    does not capture.
    """
    import joblib

    artifact_dir.mkdir(parents=True, exist_ok=True)
    native = artifact_dir / f"{stem}.json"
    model.get_booster().save_model(str(native))

    bundle_path = artifact_dir / f"{stem}_bundle.joblib"
    joblib.dump({"model": model, **bundle}, bundle_path)

    meta_path = artifact_dir / f"{stem}_metadata.json"
    serialisable = {k: v for k, v in bundle.items() if k != "model"}
    meta_path.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")

    return {"native": native, "bundle": bundle_path, "metadata": meta_path}


def feature_importance_frame(model, feature_names: list[str]) -> pd.DataFrame:
    """Gain-based importance, aligned to feature names and sorted."""
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    rows = [{"feature": f, "gain": float(scores.get(f, 0.0))} for f in feature_names]
    out = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    total = out["gain"].sum()
    out["gain_pct"] = (100 * out["gain"] / total).round(2) if total else 0.0
    return out
