"""Train an XGBoost regressor for PERM processing time.

    python -m src.models.train_regressor

Target: ``processing_days`` = decision_date - filing_date. Evaluated on the
temporal test set produced by ``src.data_prep.build_dataset`` -- never a random
split, and never the rows the model trained on.

Early stopping uses a validation slice carved off the **end of the training
period**, not the test set. Using the test set to choose the stopping round
would tune a hyperparameter on the data the score is reported from.
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

from . import evaluate as ev
from . import tracking as tk
from .data import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    DataError,
    default_tracking_uri,
    load_modelling_data,
    make_validation_split,
)

EXPERIMENT = "perm-processing-time"
TARGET = "processing_days"

DEFAULT_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    reg_alpha=0.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    random_state=42,
    n_jobs=-1,
)


def train(args: argparse.Namespace) -> dict:
    data = load_modelling_data(TARGET, data_dir=args.data_dir,
                               drop_split_year=not args.keep_year_feature)

    fit_mask, val_mask = make_validation_split(
        data, strategy=args.val_strategy, val_years=args.val_years)
    X_fit, y_fit = data.X_train[fit_mask], data.y_train[fit_mask]
    X_val, y_val = data.X_train[val_mask], data.y_train[val_mask]

    # Log-space training tames the right tail; predictions are inverted before
    # any metric is computed, so every number below is in days either way.
    y_fit_t = np.log1p(y_fit) if args.log_target else y_fit
    y_val_t = np.log1p(y_val) if args.log_target else y_val

    params = dict(DEFAULT_PARAMS)
    params.update(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    model = xgb.XGBRegressor(**params)
    started = time.time()
    model.fit(X_fit, y_fit_t, eval_set=[(X_val, y_val_t)], verbose=False)
    train_seconds = time.time() - started

    def predict(X: pd.DataFrame) -> np.ndarray:
        raw = model.predict(X)
        out = np.expm1(raw) if args.log_target else raw
        return np.clip(out, 1, None)

    pred_test = predict(data.X_test)
    pred_train = predict(data.X_train)

    # Baseline: the training median, the best constant guess available.
    baseline_value = float(np.median(data.y_train))
    baseline_test = np.full(len(data.y_test), baseline_value)

    metrics = {
        **ev.regression_metrics(data.y_test, pred_test, prefix="test_"),
        **ev.regression_metrics(data.y_train, pred_train, prefix="train_"),
        **ev.regression_metrics(data.y_test, baseline_test, prefix="baseline_"),
        "best_iteration": float(getattr(model, "best_iteration", 0) or 0),
        "train_seconds": train_seconds,
        "n_features": len(data.feature_names),
        "n_train": data.train_rows,
        "n_test": data.test_rows,
    }

    importance = tk.feature_importance_frame(model, data.feature_names)

    # ---------------- MLflow ------------------------------------------------
    uri = tk.setup_mlflow(EXPERIMENT, args.tracking_uri)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags({
            "task": "regression", "target": TARGET, "model": "xgboost",
            "split_on": data.metadata.get("split", {}).get("split_on", "?"),
            "data_dir": str(args.data_dir),
        })
        mlflow.log_params({
            **{k: v for k, v in params.items() if k != "n_jobs"},
            "log_target": args.log_target,
            "val_years": args.val_years,
            "val_strategy": args.val_strategy,
            "keep_year_feature": args.keep_year_feature,
            "excluded_features": ",".join(data.excluded_features) or "none",
            "n_features": len(data.feature_names),
            "n_categorical": len(data.categorical_features),
            "train_years": str(data.metadata.get("split", {}).get("train_years")),
            "test_years": str(data.metadata.get("split", {}).get("test_years")),
            "source_files": str(data.metadata.get("source_files")),
        })
        mlflow.log_metrics(metrics)

        tk.log_dataframe(importance, "feature_importance.csv")
        tk.log_json({"features": data.feature_names,
                     "categorical": data.categorical_features,
                     "split": data.metadata.get("split", {})}, "feature_spec.json")
        tk.log_and_save_figure(
            tk.importance_figure(importance, "Processing time - gain importance"),
            "feature_importance.png",
            args.report_dir / "regressor_feature_importance.png" if args.save_figures else None,
        )
        tk.log_and_save_figure(
            tk.residual_figure(data.y_test, pred_test,
                               "Processing time - temporal test set"),
            "residuals.png",
            args.report_dir / "regressor_residuals.png" if args.save_figures else None,
        )
        mlflow.xgboost.log_model(model, name="model")
        run_id = run.info.run_id

    paths = tk.save_model_bundle(
        model,
        {"task": "regression", "target": TARGET, "features": data.feature_names,
         "categorical_features": data.categorical_features, "categories": data.categories,
         "log_target": args.log_target, "params": params, "metrics": metrics,
         "split": data.metadata.get("split", {}), "mlflow_run_id": run_id},
        args.artifact_dir, "processing_time_xgb",
    )

    if args.preview:
        _print_report(data, metrics, importance, baseline_value, pred_test,
                      baseline_test, paths, run_id, uri, args)
    return {"metrics": metrics, "run_id": run_id, "paths": paths}


def _print_report(data, metrics, importance, baseline_value, pred_test,
                  baseline_test, paths, run_id, uri, args) -> None:
    bar = "=" * 78
    print(bar)
    print("XGBOOST REGRESSOR - PERM PROCESSING TIME")
    print(bar)
    print(f"\ndata      : {args.data_dir}")
    print(f"split     : {data.split_description}")
    print(f"train     : {data.train_rows:,} rows   test: {data.test_rows:,} rows")
    print(f"features  : {len(data.feature_names)} "
          f"({len(data.categorical_features)} categorical, native XGBoost handling)")
    if data.excluded_features:
        print(f"excluded  : {', '.join(data.excluded_features)} "
              f"(cannot extrapolate to the unseen test year)")
    print(f"best iter : {int(metrics['best_iteration'])} "
          f"(early stopping, {args.val_strategy} hold-out within train)")
    print(f"fit time  : {metrics['train_seconds']:.1f}s")

    print("\nTEST-SET PERFORMANCE (temporal hold-out)")
    print(ev.regression_report(data.y_test, pred_test, baseline_test, TARGET))
    print(f"\n  baseline  = always predict the training median "
          f"({baseline_value:.0f} days)")
    print(f"  train RMSE {metrics['train_rmse']:.1f} vs test RMSE "
          f"{metrics['test_rmse']:.1f}  "
          f"(gap {metrics['test_rmse'] - metrics['train_rmse']:+.1f} days)")

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
    p = argparse.ArgumentParser(description="Train XGBoost regressor for processing time.")
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
    p.add_argument("--val-years", type=int, default=1,
                   help="training cohorts held out for early stopping")
    p.add_argument("--val-strategy", choices=["random", "temporal"],
                   default="random",
                   help="early-stopping hold-out; see data.make_validation_split")
    p.add_argument("--keep-year-feature", action="store_true",
                   help="keep the split-year column as a feature; it cannot "
                        "extrapolate to the unseen test year (see data.py)")
    p.add_argument("--log-target", action="store_true",
                   help="train on log1p(days); metrics stay in days")
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
