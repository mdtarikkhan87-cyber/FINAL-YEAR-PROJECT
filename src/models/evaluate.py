"""Metrics and their baselines.

Every metric is reported next to a trivial baseline -- predict the training
median for regression, the training majority class for classification. On this
problem that is not a formality: the outcome target is ~88% certified, so a model
that predicts "certified" for every case scores 88% accuracy while being useless.
An accuracy figure quoted without its baseline says nothing about whether the
model learned anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}medae": float(median_absolute_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}mape": float(np.mean(np.abs(err / np.clip(y_true, 1, None))) * 100),
        f"{prefix}bias": float(np.mean(err)),
        f"{prefix}within_30d": float(np.mean(np.abs(err) <= 30) * 100),
        f"{prefix}within_60d": float(np.mean(np.abs(err) <= 60) * 100),
    }


def regression_report(y_true, y_pred, baseline_pred, target_name: str = "processing_days") -> str:
    model = regression_metrics(y_true, y_pred)
    base = regression_metrics(y_true, baseline_pred)
    lines = [
        f"  {'metric':<14}{'model':>12}{'baseline':>12}{'improvement':>14}",
        f"  {'-' * 52}",
    ]
    for key, label, unit, lower_better in [
        ("rmse", "RMSE", "days", True),
        ("mae", "MAE", "days", True),
        ("medae", "MedAE", "days", True),
        ("mape", "MAPE", "%", True),
        ("r2", "R2", "", False),
    ]:
        m, b = model[key], base[key]
        if lower_better:
            imp = f"{100 * (b - m) / abs(b):+.1f}%" if b else "n/a"
        else:
            imp = f"{m - b:+.3f}"
        lines.append(f"  {label:<14}{m:>12.3f}{b:>12.3f}{imp:>14}")
    lines += [
        f"  {'-' * 52}",
        f"  within +/-30 days : {model['within_30d']:.1f}% of cases "
        f"(baseline {base['within_30d']:.1f}%)",
        f"  within +/-60 days : {model['within_60d']:.1f}% of cases "
        f"(baseline {base['within_60d']:.1f}%)",
        f"  mean signed error : {model['bias']:+.1f} days "
        f"({'over' if model['bias'] > 0 else 'under'}-predicting on average)",
        f"  target            : {target_name}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_proba=None, labels=None,
                           prefix: str = "") -> dict[str, float]:
    out = {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None and labels is not None:
        try:
            out[f"{prefix}log_loss"] = float(
                log_loss(y_true, y_proba, labels=list(range(len(labels)))))
        except ValueError:
            pass
        try:
            out[f"{prefix}roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                              labels=list(range(len(labels)))))
        except ValueError:
            pass
    return out


def per_class_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    idx = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, zero_division=0
    )
    return pd.DataFrame({
        "class": labels,
        "support": support,
        "precision": precision.round(4),
        "recall": recall.round(4),
        "f1": f1.round(4),
    })


def confusion_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return pd.DataFrame(cm,
                        index=[f"true_{c}" for c in labels],
                        columns=[f"pred_{c}" for c in labels])


def classification_report_text(y_true, y_pred, y_proba, labels: list[str],
                               baseline_pred) -> str:
    model = classification_metrics(y_true, y_pred, y_proba, labels)
    base = classification_metrics(y_true, baseline_pred)
    lines = [
        f"  {'metric':<20}{'model':>12}{'baseline':>12}{'delta':>12}",
        f"  {'-' * 56}",
    ]
    for key, label in [
        ("accuracy", "accuracy"),
        ("balanced_accuracy", "balanced acc"),
        ("f1_macro", "F1 (macro)"),
        ("f1_weighted", "F1 (weighted)"),
    ]:
        m, b = model[key], base.get(key, float("nan"))
        lines.append(f"  {label:<20}{m:>12.4f}{b:>12.4f}{m - b:>+12.4f}")
    for key, label in [("log_loss", "log loss"), ("roc_auc_ovr", "ROC AUC (OvR)")]:
        if key in model:
            lines.append(f"  {label:<20}{model[key]:>12.4f}{'-':>12}{'-':>12}")
    lines += [
        f"  {'-' * 56}",
        "  baseline = always predict the training majority class",
        "",
        "  PER CLASS",
    ]
    pc = per_class_frame(y_true, y_pred, labels)
    for line in pc.to_string(index=False).splitlines():
        lines.append(f"    {line}")
    lines += ["", "  CONFUSION MATRIX (rows = truth, cols = prediction)"]
    cf = confusion_frame(y_true, y_pred, labels)
    for line in cf.to_string().splitlines():
        lines.append(f"    {line}")
    return "\n".join(lines)


def full_classification_report(y_true, y_pred, labels: list[str]) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(labels))),
        target_names=labels, zero_division=0, digits=4,
    )
