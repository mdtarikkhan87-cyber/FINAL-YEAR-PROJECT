"""Train an XGBoost classifier for PERM case outcome.

    python -m src.models.train_classifier

Target: ``outcome``. The prep layer produces four classes -- certified, denied,
withdrawn, certified_expired -- because that is what the disclosure data holds.
``--merge-expired`` folds certified_expired into certified for the three-class
problem, which is defensible: DOL *did* certify those cases and the employer
then let the certification lapse. Which framing the write-up uses is a stated
choice, not a default; see HANDOVER.md.

The class mix is roughly 88/6/4/2, so accuracy is close to meaningless on its
own -- predicting "certified" every time scores ~88%. Macro-F1, balanced
accuracy, and the per-class recalls are the numbers that matter, and every one
of them is printed next to the majority-class baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from . import evaluate as ev
from . import tracking as tk
from .data import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    default_tracking_uri,
    DataError,
    load_modelling_data,
    make_validation_split,
)

EXPERIMENT = "perm-outcome"
TARGET = "outcome"

DEFAULT_PARAMS = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    random_state=42,
    n_jobs=-1,
)


def train(args: argparse.Namespace) -> dict:
    data = load_modelling_data(TARGET, data_dir=args.data_dir,
                               drop_split_year=not args.keep_year_feature)

    y_train_raw = data.y_train.astype(str)
    y_test_raw = data.y_test.astype(str)
    if args.merge_expired:
        y_train_raw = y_train_raw.replace({"certified_expired": "certified"})
        y_test_raw = y_test_raw.replace({"certified_expired": "certified"})

    encoder = LabelEncoder().fit(y_train_raw)
    labels = encoder.classes_.tolist()

    unseen = set(y_test_raw.unique()) - set(labels)
    if unseen:
        raise DataError(
            f"test set contains outcome value(s) never seen in training: {sorted(unseen)}. "
            "The model cannot predict a class it was not trained on."
        )
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    fit_mask, val_mask = make_validation_split(
        data, strategy=args.val_strategy, val_years=args.val_years)
    X_fit, X_val = data.X_train[fit_mask], data.X_train[val_mask]
    y_fit, y_val = y_train[fit_mask], y_train[val_mask]

    # Without balancing, the rare classes are simply never predicted. With it,
    # macro-F1 improves at the cost of overall accuracy -- the trade is explicit.
    sample_weight = (
        compute_sample_weight("balanced", y_fit) if args.balanced else None
    )

    params = dict(DEFAULT_PARAMS)
    params.update(
        num_class=len(labels),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    model = xgb.XGBClassifier(**params)
    started = time.time()
    model.fit(X_fit, y_fit, sample_weight=sample_weight,
              eval_set=[(X_val, y_val)], verbose=False)
    train_seconds = time.time() - started

    proba_test = model.predict_proba(data.X_test)
    pred_test = proba_test.argmax(axis=1)
    pred_train = model.predict(data.X_train)

    majority = int(pd.Series(y_train).value_counts().idxmax())
    baseline_test = np.full(len(y_test), majority)

    metrics = {
        **ev.classification_metrics(y_test, pred_test, proba_test, labels, prefix="test_"),
        **ev.classification_metrics(y_train, pred_train, prefix="train_"),
        **ev.classification_metrics(y_test, baseline_test, prefix="baseline_"),
        "best_iteration": float(getattr(model, "best_iteration", 0) or 0),
        "train_seconds": train_seconds,
        "n_classes": len(labels),
        "n_features": len(data.feature_names),
        "n_train": data.train_rows,
        "n_test": data.test_rows,
    }
    per_class = ev.per_class_frame(y_test, pred_test, labels)
    for _, row in per_class.iterrows():
        for m in ("precision", "recall", "f1"):
            metrics[f"test_{m}_{row['class']}"] = float(row[m])

    confusion = ev.confusion_frame(y_test, pred_test, labels)
    importance = tk.feature_importance_frame(model, data.feature_names)

    # ---------------- MLflow ------------------------------------------------
    uri = tk.setup_mlflow(EXPERIMENT, args.tracking_uri)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags({
            "task": "classification", "target": TARGET, "model": "xgboost",
            "classes": ",".join(labels),
            "split_on": data.metadata.get("split", {}).get("split_on", "?"),
            "balanced": str(args.balanced),
            "data_dir": str(args.data_dir),
        })
        mlflow.log_params({
            **{k: v for k, v in params.items() if k != "n_jobs"},
            "merge_expired": args.merge_expired,
            "balanced": args.balanced,
            "val_years": args.val_years,
            "val_strategy": args.val_strategy,
            "keep_year_feature": args.keep_year_feature,
            "excluded_features": ",".join(data.excluded_features) or "none",
            "n_features": len(data.feature_names),
            "n_categorical": len(data.categorical_features),
            "class_labels": ",".join(labels),
            "train_years": str(data.metadata.get("split", {}).get("train_years")),
            "test_years": str(data.metadata.get("split", {}).get("test_years")),
            "source_files": str(data.metadata.get("source_files")),
        })
        mlflow.log_metrics(metrics)

        tk.log_dataframe(confusion, "confusion_matrix.csv", index=True)
        tk.log_dataframe(per_class, "per_class_metrics.csv")
        tk.log_dataframe(importance, "feature_importance.csv")
        mlflow.log_text(
            ev.full_classification_report(y_test, pred_test, labels),
            "classification_report.txt",
        )
        tk.log_json({"features": data.feature_names,
                     "categorical": data.categorical_features,
                     "classes": labels,
                     "split": data.metadata.get("split", {})}, "feature_spec.json")
        tk.log_and_save_figure(
            tk.confusion_figure(confusion, labels, "Case outcome - temporal test set"),
            "confusion_matrix.png",
            args.report_dir / "classifier_confusion_matrix.png" if args.save_figures else None,
        )
        tk.log_and_save_figure(
            tk.importance_figure(importance, "Case outcome - gain importance"),
            "feature_importance.png",
            args.report_dir / "classifier_feature_importance.png" if args.save_figures else None,
        )
        mlflow.xgboost.log_model(model, name="model")
        run_id = run.info.run_id

    paths = tk.save_model_bundle(
        model,
        {"task": "classification", "target": TARGET, "features": data.feature_names,
         "categorical_features": data.categorical_features, "categories": data.categories,
         "classes": labels, "merge_expired": args.merge_expired,
         "balanced": args.balanced, "params": params, "metrics": metrics,
         "split": data.metadata.get("split", {}), "mlflow_run_id": run_id},
        args.artifact_dir, "outcome_xgb",
    )

    if args.preview:
        _print_report(data, metrics, labels, y_test, pred_test, proba_test,
                      baseline_test, importance, paths, run_id, uri, args)
    return {"metrics": metrics, "run_id": run_id, "paths": paths}


def _print_report(data, metrics, labels, y_test, pred_test, proba_test,
                  baseline_test, importance, paths, run_id, uri, args) -> None:
    bar = "=" * 78
    print(bar)
    print("XGBOOST CLASSIFIER - PERM CASE OUTCOME")
    print(bar)
    print(f"\ndata      : {args.data_dir}")
    print(f"split     : {data.split_description}")
    print(f"train     : {data.train_rows:,} rows   test: {data.test_rows:,} rows")
    print(f"features  : {len(data.feature_names)} "
          f"({len(data.categorical_features)} categorical)")
    if data.excluded_features:
        print(f"excluded  : {', '.join(data.excluded_features)} "
              f"(cannot extrapolate to the unseen test year)")
    print(f"classes   : {len(labels)} -> {', '.join(labels)}"
          f"{'  (certified_expired merged into certified)' if args.merge_expired else ''}")
    print(f"balanced  : {args.balanced}"
          f"{'' if args.balanced else '  (rare classes will be under-predicted)'}")
    print(f"best iter : {int(metrics['best_iteration'])}   "
          f"fit time: {metrics['train_seconds']:.1f}s")

    print("\nCLASS BALANCE (train)")
    mix = data.y_train.astype(str)
    if args.merge_expired:
        mix = mix.replace({"certified_expired": "certified"})
    dist = (mix.value_counts(normalize=True) * 100).round(2)
    for cls, pct in dist.items():
        print(f"  {cls:<20}{pct:>7.2f}%")

    print("\nTEST-SET PERFORMANCE (temporal hold-out)")
    print(ev.classification_report_text(y_test, pred_test, proba_test, labels, baseline_test))

    print("\nTOP 12 FEATURES BY GAIN")
    top = importance.head(12).copy()
    top["gain"] = top["gain"].round(1)
    print("  " + top.to_string(index=False).replace("\n", "\n  "))

    print(f"\nMLFLOW\n  experiment : {EXPERIMENT}\n  run_id     : {run_id}"
          f"\n  tracking   : {uri}")
    print("\nARTIFACTS")
    for label, path in paths.items():
        print(f"  {label:<9} {path}")
    print(bar)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train XGBoost classifier for case outcome.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI (default: local sqlite mlflow.db)")
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--run-name", default=None)
    p.add_argument("--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"])
    p.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    p.add_argument("--max-depth", type=int, default=DEFAULT_PARAMS["max_depth"])
    p.add_argument("--early-stopping-rounds", type=int, default=75)
    p.add_argument("--val-years", type=int, default=1)
    p.add_argument("--val-strategy", choices=["random", "temporal"],
                   default="random",
                   help="early-stopping hold-out; see data.make_validation_split")
    p.add_argument("--keep-year-feature", action="store_true",
                   help="keep the split-year column as a feature; it cannot "
                        "extrapolate to the unseen test year (see data.py)")
    p.add_argument("--merge-expired", action="store_true",
                   help="fold certified_expired into certified (3-class problem)")
    p.add_argument("--balanced", action="store_true",
                   help="apply balanced sample weights; trades accuracy for macro-F1")
    p.add_argument("--no-figures", dest="save_figures", action="store_false")
    p.add_argument("--no-preview", dest="preview", action="store_false")
    args = p.parse_args(argv)
    if args.tracking_uri is None:
        args.tracking_uri = default_tracking_uri()

    try:
        train(args)
    except DataError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
