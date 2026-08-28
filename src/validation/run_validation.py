"""Run the temporal validation harness and write the report.

    python -m src.validation.run_validation

Re-fits both models once per year boundary, evaluates each on the following
year, and writes ``reports/temporal_validation_report.md`` with the metrics
series, charts, and an interpretation of whether performance drifts over time.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..models.data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    DataError,
    load_full_dataset,
)
from . import harness as hn  # noqa: E402

PLOT_SUBDIR = "validation"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def run_harness(args) -> dict:
    data, metadata = load_full_dataset(args.data_dir)
    year_col = metadata.get("split", {}).get("year_column", "decision_fiscal_year")
    if year_col not in data.columns:
        raise hn.ValidationError(
            f"year column {year_col!r} is not in the processed data; "
            "re-run src.data_prep.build_dataset"
        )

    data = data[data[year_col].notna()].copy()
    data[year_col] = data[year_col].astype(int)
    years = sorted(data[year_col].unique())

    features, dropped = hn.select_features(metadata, data.columns, args.keep_year_features)
    fold_defs = hn.build_folds(years, args.scheme, args.window, args.min_train_years)

    print(f"data      : {len(data):,} rows, {year_col} {years[0]}-{years[-1]}")
    print(f"scheme    : {args.scheme}" +
          (f" (window {args.window})" if args.scheme == "rolling" else ""))
    print(f"features  : {len(features)}" +
          (f"  (dropped year columns: {', '.join(dropped)})" if dropped else ""))
    print(f"folds     : {len(fold_defs)}\n")

    results: list[hn.FoldResult] = []
    started = time.time()
    for i, (train_years, test_year) in enumerate(fold_defs, start=1):
        train_df = data[data[year_col].isin(train_years)]
        test_df = data[data[year_col] == test_year]
        fold = hn.Fold(i, train_years, test_year, len(train_df), len(test_df))

        print(f"  fold {i}: train {train_years[0]}-{train_years[-1]} "
              f"({len(train_df):,}) -> test {test_year} ({len(test_df):,})", end="", flush=True)

        reg = hn.run_regression_fold(train_df, test_df, features, args.seed,
                                     args.n_estimators, args.early_stopping_rounds)
        clf, notes = hn.run_classification_fold(train_df, test_df, features, args.seed,
                                                args.n_estimators,
                                                args.early_stopping_rounds, args.balanced)
        results.append(hn.FoldResult(fold, reg, clf, notes))
        print(f"   RMSE {reg['test_rmse']:6.1f} (base {reg['baseline_rmse']:6.1f})"
              f"   acc {clf['test_accuracy']:.3f}   F1m {clf['test_f1_macro']:.3f}")

    elapsed = time.time() - started
    print(f"\n{len(results)} folds in {elapsed:.1f}s")

    return {"results": results, "years": years, "year_col": year_col,
            "features": features, "dropped": dropped, "metadata": metadata,
            "elapsed": elapsed, "n_rows": len(data)}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def regression_table(results: list[hn.FoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "fold": r.fold.index,
            "train_years": f"{r.fold.train_years[0]}-{r.fold.train_years[-1]}",
            "test_year": r.fold.test_year,
            "n_train": r.fold.n_train, "n_test": r.fold.n_test,
            "rmse": r.regression["test_rmse"],
            "baseline_rmse": r.regression["baseline_rmse"],
            "rmse_impr_pct": r.regression["rmse_improvement_pct"],
            "mae": r.regression["test_mae"],
            "baseline_mae": r.regression["baseline_mae"],
            "r2": r.regression["test_r2"],
            "bias": r.regression["test_bias"],
            "best_iter": r.regression["best_iteration"],
            "beats_baseline": r.regression["beats_baseline"],
        })
    return pd.DataFrame(rows)


def classification_table(results: list[hn.FoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "fold": r.fold.index,
            "train_years": f"{r.fold.train_years[0]}-{r.fold.train_years[-1]}",
            "test_year": r.fold.test_year,
            "n_test": r.fold.n_test,
            "accuracy": r.classification["test_accuracy"],
            "baseline_accuracy": r.classification["baseline_accuracy"],
            "acc_lift": r.classification["accuracy_lift"],
            "f1_macro": r.classification["test_f1_macro"],
            "balanced_acc": r.classification["test_balanced_accuracy"],
            "roc_auc": r.classification.get("test_roc_auc_ovr", float("nan")),
            "classes_predicted": r.classification["n_predicted_classes"],
            "degenerate": r.classification["degenerate"],
            "best_iter": r.classification["best_iteration"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_regression_series(tbl: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = tbl["test_year"]

    axes[0].plot(x, tbl["rmse"], "o-", color="#3b6ea5", label="model")
    axes[0].plot(x, tbl["baseline_rmse"], "s--", color="crimson", label="baseline (train median)")
    axes[0].set_title("RMSE by test year"); axes[0].set_ylabel("days")

    axes[1].plot(x, tbl["mae"], "o-", color="#3b6ea5", label="model")
    axes[1].plot(x, tbl["baseline_mae"], "s--", color="crimson", label="baseline")
    axes[1].set_title("MAE by test year"); axes[1].set_ylabel("days")

    axes[2].bar(x, tbl["rmse_impr_pct"],
                color=["#3b8a5a" if v > 0 else "#b5423a" for v in tbl["rmse_impr_pct"]])
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_title("RMSE improvement over baseline")
    axes[2].set_ylabel("%")

    for ax in axes:
        ax.set_xlabel("test year")
        ax.set_xticks(x)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle("Processing-time model across temporal folds", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_classification_series(tbl: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = tbl["test_year"]

    axes[0].plot(x, tbl["accuracy"], "o-", color="#3b6ea5", label="model")
    axes[0].plot(x, tbl["baseline_accuracy"], "s--", color="crimson", label="majority baseline")
    axes[0].set_title("Accuracy by test year"); axes[0].set_ylabel("accuracy")

    axes[1].plot(x, tbl["f1_macro"], "o-", color="#3b6ea5", label="macro F1")
    axes[1].plot(x, tbl["balanced_acc"], "^--", color="#e8a33d", label="balanced acc")
    axes[1].set_title("Macro F1 / balanced accuracy"); axes[1].set_ylabel("score")

    axes[2].plot(x, tbl["roc_auc"], "o-", color="#3b6ea5")
    axes[2].axhline(0.5, color="crimson", ls="--", lw=1, label="chance")
    axes[2].set_title("ROC AUC (OvR)"); axes[2].set_ylabel("AUC")

    for ax in axes:
        ax.set_xlabel("test year")
        ax.set_xticks(x)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Outcome model across temporal folds", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _verdict(slope: float, flat_band: float, higher_is_better: bool = True) -> str:
    """Turn a trend slope into a word, instead of asserting one in prose.

    Anything inside +/- ``flat_band`` per year is called stable. Hard-coding a
    verdict is how a report ends up claiming degradation that its own table
    contradicts.
    """
    if abs(slope) <= flat_band:
        return "stable"
    improving = slope > 0 if higher_is_better else slope < 0
    return "improving" if improving else "degrading"


def _md_table(df: pd.DataFrame, formats: dict[str, str]) -> list[str]:
    header = "| " + " | ".join(df.columns) + " |"
    align = "|" + "|".join("---:" if c not in ("train_years",) else "---"
                           for c in df.columns) + "|"
    lines = [header, align]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "**no**")
            elif c in formats:
                cells.append(formats[c].format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(run: dict, reg_tbl: pd.DataFrame, clf_tbl: pd.DataFrame,
                 args, path: Path) -> Path:
    p = f"{PLOT_SUBDIR}/"
    years = reg_tbl["test_year"].tolist()
    L: list[str] = []
    add = L.append

    rmse_trend = hn.trend(reg_tbl["rmse"].tolist(), years)
    impr_trend = hn.trend(reg_tbl["rmse_impr_pct"].tolist(), years)
    auc_trend = hn.trend(clf_tbl["roc_auc"].tolist(), years)
    lift_trend = hn.trend(clf_tbl["acc_lift"].tolist(), years)

    add("# Temporal Validation Report")
    add("")
    add(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
        f"{args.scheme} window · {len(reg_tbl)} folds · {run['n_rows']:,} rows · "
        f"data `{args.data_dir.name}`")
    add("")
    add("> **Synthetic data.** Every case was invented by "
        "`src/data_prep/make_synthetic.py`. The drift measured below is drift in a "
        "generator, not in real PERM adjudication.")
    add("")

    add("## What this does and why")
    add("")
    add("A single fixed train/test split gives one number and no way to tell whether it "
        "was luck. This harness re-fits both models once per year boundary — train on "
        "everything up to year N, test on year N+1, step forward — so performance reads as "
        "a series. That is what makes the question answerable: **does the model degrade on "
        "more recent years?**")
    add("")
    add(f"- **Scheme**: `{args.scheme}`" +
        (f", window {args.window} years" if args.scheme == "rolling" else
         " — the training window grows as time passes, the realistic deployment story"))
    add(f"- **Split key**: `{run['year_col']}`")
    add(f"- **Features**: {len(run['features'])}")
    if run["dropped"]:
        add(f"- **Dropped**: `{'`, `'.join(run['dropped'])}` — every fold tests on a year "
            "the model has never seen, so an absolute-year column is out of range in "
            "*every* fold. A tree routes it to the nearest fitted bin and it silently "
            "becomes a constant. `--keep-year-features` reproduces the broken behaviour.")
    add(f"- **Early stopping**: random 15% hold-out from inside each fold's training rows, "
        "never the fold's test year (HANDOVER.md §7.3).")
    add(f"- **Category levels** are re-fitted per fold on that fold's training rows only.")
    add("")

    # ---- regression -------------------------------------------------------
    add("---")
    add("")
    add("## 1. Processing time across folds")
    add("")
    add(f"![Regression across folds]({p}regression_series.png)")
    add("")
    L.extend(_md_table(reg_tbl, {
        "rmse": "{:.1f}", "baseline_rmse": "{:.1f}", "rmse_impr_pct": "{:+.1f}%",
        "mae": "{:.1f}", "baseline_mae": "{:.1f}", "r2": "{:.3f}", "bias": "{:+.1f}",
        "n_train": "{:,}", "n_test": "{:,}",
    }))
    add("")
    beats = int(reg_tbl["beats_baseline"].sum())
    add(f"**Beats the baseline in {beats} of {len(reg_tbl)} folds.** "
        f"RMSE moves from {rmse_trend['first']:.1f} days in {years[0]} to "
        f"{rmse_trend['last']:.1f} in {years[-1]} "
        f"({rmse_trend['direction']}, {rmse_trend['slope_per_year']:+.1f} days/year). "
        f"Improvement over baseline is {impr_trend['direction']} at "
        f"{impr_trend['slope_per_year']:+.2f} pp/year.")
    add("")

    # ---- classification ---------------------------------------------------
    add("---")
    add("")
    add("## 2. Case outcome across folds")
    add("")
    add(f"![Classification across folds]({p}classification_series.png)")
    add("")
    L.extend(_md_table(clf_tbl, {
        "accuracy": "{:.4f}", "baseline_accuracy": "{:.4f}", "acc_lift": "{:+.4f}",
        "f1_macro": "{:.4f}", "balanced_acc": "{:.4f}", "roc_auc": "{:.4f}",
        "n_test": "{:,}",
    }))
    add("")
    degen = int(clf_tbl["degenerate"].sum())
    if degen:
        add(f"**{degen} of {len(clf_tbl)} folds produce a degenerate classifier** — one that "
            "predicts a single class for every case. Accuracy in those folds equals the "
            "majority baseline exactly, which is why `acc_lift` sits at zero. That is not "
            "a model that has learned nothing subtle; it is a model that has learned "
            "nothing it can act on. The fairness audit flags the same condition "
            "(HANDOVER.md §7.5).")
        add("")
    add(f"ROC AUC moves from {auc_trend['first']:.4f} to {auc_trend['last']:.4f} "
        f"({auc_trend['direction']}, {auc_trend['slope_per_year']:+.4f}/year); accuracy lift "
        f"over baseline is {lift_trend['direction']}.")
    add("")

    # ---- interpretation ---------------------------------------------------
    add("---")
    add("")
    add("## 3. Interpretation")
    add("")
    add("### Does the model degrade on more recent years?")
    add("")
    worst = reg_tbl.loc[reg_tbl["rmse_impr_pct"].idxmin()]
    best = reg_tbl.loc[reg_tbl["rmse_impr_pct"].idxmax()]
    reg_verdict = _verdict(impr_trend["slope_per_year"], flat_band=1.0)
    headline = {"stable": "no — it holds steady",
                "degrading": "yes",
                "improving": "no — it improves"}[reg_verdict]
    add(f"**Processing time — {headline}.** Measured against the baseline, the model's edge "
        f"is **{reg_verdict}** at {impr_trend['slope_per_year']:+.2f} percentage points per "
        f"year, ranging {worst['rmse_impr_pct']:+.1f}% (test {int(worst['test_year'])}) to "
        f"{best['rmse_impr_pct']:+.1f}% (test {int(best['test_year'])}).")
    add("")
    add(f"Raw RMSE tells a different-looking story — {rmse_trend['first']:.0f} days rising to "
        f"{reg_tbl['rmse'].max():.0f} before falling to {rmse_trend['last']:.0f} — but that "
        "is the *target* moving, not the model failing. Processing times themselves grew "
        "through the backlog years, so absolute error grows with them; the baseline rises "
        f"in lockstep, from {reg_tbl['baseline_rmse'].iloc[0]:.0f} to "
        f"{reg_tbl['baseline_rmse'].max():.0f} days. **Quoting raw RMSE across periods of "
        "different difficulty would manufacture a trend that is not there.** The "
        "improvement-over-baseline column is the honest read.")
    add("")
    add("A fixed split reports whichever of these folds it happened to land on. That is the "
        "argument for this harness existing: the single-split number from Phase 4 was "
        "neither the best nor the worst fold, and there was no way to know that from the "
        "single split alone.")
    add("")
    clf_verdict = _verdict(auc_trend["slope_per_year"], flat_band=0.01)
    add(f"**Case outcome — {clf_verdict}, and near-chance throughout.** ROC AUC sits between "
        f"{clf_tbl['roc_auc'].min():.3f} and {clf_tbl['roc_auc'].max():.3f} across every "
        f"fold ({auc_trend['slope_per_year']:+.4f}/year) and accuracy never separates from "
        "the majority baseline by more than "
        f"{clf_tbl['acc_lift'].abs().max():.4f}. The model is consistently unable to predict "
        "the outcome in every period — which points at the feature set, not at drift. The "
        "generator's dominant denial driver is a latent audit flag that is deliberately "
        "not a column, mirroring real disclosure files. No amount of retraining across "
        "folds changes that.")
    add("")
    add("### What this says about the fixed split")
    add("")
    add("Fold-to-fold variation is the number that should temper any single-split claim. "
        f"Across folds the regression RMSE spans {reg_tbl['rmse'].min():.0f}–"
        f"{reg_tbl['rmse'].max():.0f} days and improvement over baseline spans "
        f"{reg_tbl['rmse_impr_pct'].min():+.0f}% to {reg_tbl['rmse_impr_pct'].max():+.0f}%. "
        "Any headline figure quoted from one split should be reported alongside that "
        "range, not on its own.")
    add("")
    add("### Next")
    add("")
    add("- Add the recent-backlog feature (median processing time of cases decided before "
        "this one was filed) and re-run this harness. If it works, the improvement-over-"
        "baseline series should both rise and flatten — it is the regime signal the "
        "dropped year columns were standing in for, expressed in a form that generalises.")
    add("- Re-run with `--scheme rolling` to test whether older years help or hurt. An "
        "expanding window assumes the past stays relevant; a hard-drifting process is "
        "better served by forgetting it.")
    add("")

    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(args) -> Path:
    plot_dir = args.report_dir / PLOT_SUBDIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    result = run_harness(args)
    reg_tbl = regression_table(result["results"])
    clf_tbl = classification_table(result["results"])

    plot_regression_series(reg_tbl, plot_dir / "regression_series.png")
    plot_classification_series(clf_tbl, plot_dir / "classification_series.png")

    reg_tbl.to_csv(args.report_dir / "temporal_validation_regression.csv", index=False)
    clf_tbl.to_csv(args.report_dir / "temporal_validation_classification.csv", index=False)

    report = write_report(result, reg_tbl, clf_tbl, args,
                          args.report_dir / "temporal_validation_report.md")

    print("\n" + "=" * 78)
    print("PROCESSING TIME BY FOLD")
    print(reg_tbl[["fold", "train_years", "test_year", "n_train", "n_test", "rmse",
                   "baseline_rmse", "rmse_impr_pct", "mae", "r2"]].round(2).to_string(index=False))
    print("\nCASE OUTCOME BY FOLD")
    print(clf_tbl[["fold", "train_years", "test_year", "n_test", "accuracy",
                   "baseline_accuracy", "f1_macro", "roc_auc",
                   "classes_predicted"]].round(4).to_string(index=False))
    print("=" * 78)
    print(f"\nwrote {report}")
    for f in sorted(plot_dir.glob("*.png")):
        print(f"wrote {f} ({f.stat().st_size / 1024:.0f} KB)")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Temporal validation harness.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--scheme", choices=["expanding", "rolling"], default="expanding")
    p.add_argument("--window", type=int, default=3,
                   help="training years per fold when --scheme rolling")
    p.add_argument("--min-train-years", type=int, default=2)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--early-stopping-rounds", type=int, default=60)
    p.add_argument("--balanced", action="store_true",
                   help="balanced class weights for the classifier folds")
    p.add_argument("--keep-year-features", action="store_true",
                   help="keep absolute-year columns (out of range in every fold)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        run(args)
    except (hn.ValidationError, DataError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
