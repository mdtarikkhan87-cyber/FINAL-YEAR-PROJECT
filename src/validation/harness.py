"""Rolling / expanding-window temporal validation.

A single fixed train/test split gives one number and no way to tell whether it
was luck. This harness re-fits both models once per year boundary -- train on
everything up to year N, test on year N+1, step forward -- so performance can be
read as a series rather than a point estimate. That answers the question a
single split cannot: **is the model getting worse on more recent years?**

Two rules are enforced per fold, both learned the hard way in earlier phases:

* **Category levels are re-fitted on each fold's own training rows.** Fitting
  once over the whole dataset would leak later years' vocabulary backwards.
* **Absolute-year columns are dropped by default.** Every fold tests on a year
  the model has never seen, so a year-valued feature is out of range in *every*
  fold -- a tree routes it to the nearest fitted bin and the column silently
  becomes a constant. Phase 5 measured this costing more than the model gained
  (HANDOVER.md §7.4); ``--keep-year-features`` reproduces it for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from ..models import evaluate as ev
from ..models.data import prepare_fold
from ..models.train_classifier import DEFAULT_PARAMS as CLF_PARAMS
from ..models.train_regressor import DEFAULT_PARAMS as REG_PARAMS

# Any column holding an absolute calendar/fiscal year. Trees cannot extrapolate
# past the range they were fitted on, so under temporal validation these are
# out of range in every fold by construction.
YEAR_FEATURES = ("filing_fiscal_year", "decision_fiscal_year")

VALIDATION_FRACTION = 0.15


class ValidationError(RuntimeError):
    """Raised when a temporal validation harness cannot be built."""


@dataclass
class Fold:
    index: int
    train_years: list[int]
    test_year: int
    n_train: int
    n_test: int


@dataclass
class FoldResult:
    fold: Fold
    regression: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def build_folds(years: list[int], scheme: str = "expanding",
                window: int = 3, min_train_years: int = 2) -> list[tuple[list[int], int]]:
    """Year boundaries for each fold: (train_years, test_year).

    ``expanding`` grows the training window as time passes -- the realistic
    deployment story, where you keep every file you have ever downloaded.
    ``rolling`` keeps a fixed-length window, which trades data volume for
    recency and is the better choice if the process drifts hard.
    """
    if scheme not in {"expanding", "rolling"}:
        raise ValidationError(f"scheme must be 'expanding' or 'rolling', got {scheme!r}")
    years = sorted(years)
    folds: list[tuple[list[int], int]] = []
    for i in range(min_train_years, len(years)):
        train = years[:i] if scheme == "expanding" else years[max(0, i - window):i]
        if len(train) < min_train_years:
            continue
        folds.append((train, years[i]))
    if not folds:
        raise ValidationError(
            f"no folds could be formed from years {years} with "
            f"min_train_years={min_train_years}. Need at least "
            f"{min_train_years + 1} distinct years."
        )
    return folds


def _random_validation_mask(n: int, seed: int) -> np.ndarray:
    """Random hold-out from within the fold's training rows, for early stopping.

    Not a temporal hold-out: the last training year sits in a different backlog
    regime, so every extra tree looks worse on it and early stopping selects
    round zero (HANDOVER.md §7.3). This leaks nothing -- the fold's test year is
    untouched.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=max(1, int(n * VALIDATION_FRACTION)), replace=False)] = True
    return mask


def _fit_params(base: dict, n_estimators: int, early_stopping: int) -> dict:
    params = dict(base)
    params.update(n_estimators=n_estimators, early_stopping_rounds=early_stopping)
    return params


def run_regression_fold(train_df, test_df, feature_names, seed, n_estimators,
                        early_stopping) -> dict:
    """Fit and score the processing-time model on one fold."""
    X_train, X_test, _, _ = prepare_fold(train_df, test_df, feature_names)
    y_train = train_df["processing_days"].to_numpy(dtype=float)
    y_test = test_df["processing_days"].to_numpy(dtype=float)

    val = _random_validation_mask(len(X_train), seed)
    model = xgb.XGBRegressor(**_fit_params(REG_PARAMS, n_estimators, early_stopping))
    model.fit(X_train[~val], y_train[~val],
              eval_set=[(X_train[val], y_train[val])], verbose=False)

    pred = np.clip(model.predict(X_test), 1, None)
    baseline = np.full(len(y_test), float(np.median(y_train)))

    out = {
        **ev.regression_metrics(y_test, pred, prefix="test_"),
        **ev.regression_metrics(y_test, baseline, prefix="baseline_"),
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
        "baseline_value": float(np.median(y_train)),
    }
    out["rmse_improvement_pct"] = 100.0 * (
        out["baseline_rmse"] - out["test_rmse"]) / out["baseline_rmse"]
    out["mae_improvement_pct"] = 100.0 * (
        out["baseline_mae"] - out["test_mae"]) / out["baseline_mae"]
    out["beats_baseline"] = bool(out["test_rmse"] < out["baseline_rmse"])
    return out


def run_classification_fold(train_df, test_df, feature_names, seed, n_estimators,
                            early_stopping, balanced: bool) -> tuple[dict, list[str]]:
    """Fit and score the outcome model on one fold."""
    notes: list[str] = []
    X_train, X_test, _, _ = prepare_fold(train_df, test_df, feature_names)

    y_train_raw = train_df["outcome"].astype(str)
    y_test_raw = test_df["outcome"].astype(str)

    encoder = LabelEncoder().fit(y_train_raw)
    labels = encoder.classes_.tolist()

    # A class absent from this fold's training years cannot be predicted; those
    # test rows are excluded from the fold's metrics rather than silently scored.
    unseen = ~y_test_raw.isin(labels)
    if unseen.any():
        notes.append(
            f"{int(unseen.sum())} test rows dropped: outcome(s) "
            f"{sorted(set(y_test_raw[unseen]))} absent from this fold's training years"
        )
        X_test = X_test[~unseen.to_numpy()]
        y_test_raw = y_test_raw[~unseen]

    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    sample_weight = None
    if balanced:
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight("balanced", y_train)

    val = _random_validation_mask(len(X_train), seed)
    params = _fit_params(CLF_PARAMS, n_estimators, early_stopping)
    params["num_class"] = len(labels)
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train[~val], y_train[~val],
        sample_weight=None if sample_weight is None else sample_weight[~val],
        eval_set=[(X_train[val], y_train[val])], verbose=False,
    )

    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    majority = int(pd.Series(y_train).value_counts().idxmax())
    baseline = np.full(len(y_test), majority)

    out = {
        **ev.classification_metrics(y_test, pred, proba, labels, prefix="test_"),
        **ev.classification_metrics(y_test, baseline, prefix="baseline_"),
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
        "n_classes": len(labels),
        "n_predicted_classes": int(len(np.unique(pred))),
        "classes": ",".join(labels),
    }
    out["accuracy_lift"] = out["test_accuracy"] - out["baseline_accuracy"]
    out["f1_macro_lift"] = out["test_f1_macro"] - out["baseline_f1_macro"]
    # A one-class predictor is the degenerate case the fairness audit flags too.
    out["degenerate"] = bool(out["n_predicted_classes"] <= 1)
    return out, notes


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------
def trend(values: list[float], years: list[int]) -> dict:
    """Least-squares slope of a metric against test year, plus first/last delta.

    The slope answers the question the harness exists for: is performance
    drifting as the test year moves forward? With only a handful of folds this
    is descriptive, not inferential -- no significance is claimed.
    """
    if len(values) < 2:
        return {"slope_per_year": float("nan"), "first": float("nan"),
                "last": float("nan"), "change": float("nan"), "direction": "n/a"}
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    change = float(y[-1] - y[0])
    return {
        "slope_per_year": slope,
        "first": float(y[0]), "last": float(y[-1]), "change": change,
        "direction": "rising" if slope > 0 else ("falling" if slope < 0 else "flat"),
    }


def select_features(metadata: dict, columns, keep_year_features: bool) -> tuple[list[str], list[str]]:
    """Feature list for the harness, minus year columns unless asked otherwise."""
    declared = [c for c in metadata["feature_columns"] if c in columns]
    if keep_year_features:
        return declared, []
    dropped = [c for c in declared if c in YEAR_FEATURES]
    return [c for c in declared if c not in YEAR_FEATURES], dropped
