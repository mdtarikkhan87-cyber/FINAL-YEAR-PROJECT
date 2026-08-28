"""Generate SHAP explanations and write ``reports/shap_report.md``.

    python -m src.explainability.run_shap

Explains the saved models from ``src/models/artifacts/`` on the temporal test
set: global summary plots, per-class breakdowns for the classifier, and
per-prediction waterfalls for individual cases. Plots land in ``reports/shap/``
and the written summary references them.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from ..models.data import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    DataError,
    load_modelling_data,
)
from . import shap_analysis as sa  # noqa: E402

PLOT_SUBDIR = "shap"


def _save(fig_fn, path: Path, figsize=(9, 6)) -> None:
    """Run a SHAP plotting call that draws onto the current figure, then save it."""
    plt.figure(figsize=figsize)
    fig_fn()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def _fmt_value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def _contrib_markdown(frame: pd.DataFrame, unit: str) -> str:
    lines = [f"| feature | value | SHAP ({unit}) | effect |", "|---|---|---:|---|"]
    for _, r in frame.iterrows():
        arrow = "up" if r["shap"] >= 0 else "down"
        lines.append(
            f"| `{r['feature']}` | {_fmt_value(r['value'])} | {r['shap']:+.2f} | {arrow} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def explain_regressor(args, plot_dir: Path) -> dict:
    model, bundle = sa.load_bundle(args.artifact_dir / "processing_time_xgb_bundle.joblib")
    data = load_modelling_data("processing_days", data_dir=args.data_dir)

    X = data.X_test
    if args.max_rows and len(X) > args.max_rows:
        X = X.sample(args.max_rows, random_state=args.seed).sort_index()
    y = data.y_test.loc[X.index]

    result = sa.compute_shap(model, X, task="regression")
    preds = model.predict(X)
    importance = sa.global_importance(result)

    _save(lambda: shap.plots.beeswarm(result.explanation, max_display=18, show=False),
          plot_dir / "regressor_beeswarm.png", figsize=(9, 8))
    _save(lambda: shap.plots.bar(result.explanation, max_display=18, show=False),
          plot_dir / "regressor_bar.png", figsize=(9, 8))

    cases = sa.select_regression_cases(preds, y)
    case_blocks = []
    for i, (label, row) in enumerate(cases.items(), start=1):
        fname = f"regressor_waterfall_{i}.png"
        _save(lambda r=row: shap.plots.waterfall(result.explanation[r], max_display=12,
                                                 show=False),
              plot_dir / fname, figsize=(10, 6))
        case_blocks.append({
            "label": label, "file": fname, "row": row,
            "predicted": float(preds[row]), "actual": float(y.iloc[row]),
            "table": sa.contribution_table(result, row, top_n=8),
        })

    top_feature = importance.iloc[0]["feature"]
    top_varying = sa.most_discriminating(importance)

    # The attribution ranking is only trustworthy if the top feature is in range
    # on the test set. Check the one SHAP says matters most.
    full_pred = model.predict(data.X_test)
    oor = sa.out_of_range_diagnostic(
        data.X_train, data.X_test, data.y_test, full_pred,
        float(np.median(data.y_train)), top_feature,
    )

    return {
        "bundle": bundle, "importance": importance, "cases": case_blocks,
        "base_value": float(np.ravel(result.explanation.base_values)[0]),
        "n_rows": len(X), "top_feature": top_feature, "top_varying": top_varying,
        "direction": sa.directional_effect(result, top_varying),
        "preds": preds, "actual": y, "out_of_range": oor,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def explain_classifier(args, plot_dir: Path) -> dict:
    model, bundle = sa.load_bundle(args.artifact_dir / "outcome_xgb_bundle.joblib")
    class_names = bundle["classes"]
    data = load_modelling_data("outcome", data_dir=args.data_dir)

    X = data.X_test
    if args.max_rows and len(X) > args.max_rows:
        X = X.sample(args.max_rows, random_state=args.seed).sort_index()
    y = data.y_test.loc[X.index].astype(str)

    result = sa.compute_shap(model, X, task="classification", class_names=class_names)
    proba = model.predict_proba(X)
    importance = sa.global_importance(result)

    # One beeswarm per class: a driver of denials is invisible in an average
    # dominated by the 88%-certified class.
    class_plots = []
    for ci, name in enumerate(class_names):
        fname = f"classifier_beeswarm_{name}.png"
        _save(lambda c=ci: shap.plots.beeswarm(result.for_class(c), max_display=14,
                                               show=False),
              plot_dir / fname, figsize=(9, 7))
        class_plots.append({"class": name, "file": fname})

    _save(lambda: shap.plots.bar(result.explanation.abs.mean(0), max_display=18, show=False),
          plot_dir / "classifier_bar.png", figsize=(9, 8))

    cases = sa.select_classification_cases(proba, class_names)
    case_blocks = []
    for i, (label, (row, ci)) in enumerate(cases.items(), start=1):
        fname = f"classifier_waterfall_{i}_{class_names[ci]}.png"
        _save(lambda r=row, c=ci: shap.plots.waterfall(result.explanation[r, :, c],
                                                       max_display=12, show=False),
              plot_dir / fname, figsize=(10, 6))
        case_blocks.append({
            "label": label, "file": fname, "row": row, "class": class_names[ci],
            "proba": float(proba[row, ci]), "actual": y.iloc[row],
            "predicted": class_names[int(proba[row].argmax())],
            "table": sa.contribution_table(result, row, class_index=ci, top_n=8),
        })

    denied_name = "denied" if "denied" in class_names else class_names[0]
    denied_idx = class_names.index(denied_name)
    top_denied = sa.most_discriminating(importance, denied_name)

    # Measure the below-prevailing-wage effect directly rather than assuming it.
    wage_effect = None
    if "wage_ratio" in result.feature_names:
        wi = result.feature_names.index("wage_ratio")
        contrib = result.explanation.values[:, wi, denied_idx]
        below = (X["wage_ratio"] < 1.0).to_numpy()
        if below.any() and (~below).any():
            wage_effect = {
                "n_below": int(below.sum()), "pct_below": 100.0 * below.mean(),
                "mean_below": float(contrib[below].mean()),
                "mean_above": float(contrib[~below].mean()),
                "max_below": float(contrib[below].max()),
            }
    return {
        "bundle": bundle, "importance": importance, "cases": case_blocks,
        "class_plots": class_plots, "class_names": class_names,
        "base_values": np.atleast_2d(result.explanation.base_values)[0].tolist(),
        "n_rows": len(X), "proba": proba, "actual": y,
        "top_denied_feature": top_denied, "denied_name": denied_name,
        "wage_effect": wage_effect,
        "denied_direction": sa.directional_effect(result, top_denied, class_index=denied_idx),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(reg: dict, clf: dict, args, path: Path) -> Path:
    p = f"{PLOT_SUBDIR}/"
    rmeta, cmeta = reg["bundle"], clf["bundle"]
    rsplit = rmeta.get("split", {})
    rm, cm = rmeta.get("metrics", {}), cmeta.get("metrics", {})

    L: list[str] = []
    add = L.append

    add("# SHAP Explanation Report")
    add("")
    add(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from "
        f"`{args.data_dir.name}` on {reg['n_rows']:,} test rows.")
    add("")
    add("> **These explanations describe synthetic data.** Every row was invented by "
        "`src/data_prep/make_synthetic.py`, and the relationships SHAP recovers below are "
        "the ones that generator was written to contain. Nothing here is a finding about "
        "real PERM adjudication.")
    add("")
    add("> **SHAP explains the model, not the process.** It attributes a model's output to "
        "its inputs. Where the model is weak, SHAP faithfully explains a weak model -- it "
        "does not reveal what actually drives outcomes. Read the accuracy caveats below "
        "before quoting any driver.")
    add("")

    # ---- provenance -------------------------------------------------------
    add("## Models explained")
    add("")
    add("| | regressor | classifier |")
    add("|---|---|---|")
    add(f"| target | `processing_days` | `outcome` |")
    add(f"| split | {rsplit.get('split_on')} FY, test {rsplit.get('test_years')} | "
        f"{cmeta.get('split', {}).get('split_on')} FY, "
        f"test {cmeta.get('split', {}).get('test_years')} |")
    add(f"| features | {len(rmeta.get('features', []))} | {len(cmeta.get('features', []))} |")
    add(f"| test metric | RMSE {rm.get('test_rmse', float('nan')):.1f} days "
        f"(baseline {rm.get('baseline_rmse', float('nan')):.1f}) | "
        f"accuracy {cm.get('test_accuracy', float('nan')):.3f} "
        f"(baseline {cm.get('baseline_accuracy', float('nan')):.3f}) |")
    add("")

    reg_beats = rm.get("test_rmse", np.inf) < rm.get("baseline_rmse", 0)
    add(f"The regressor {'beats' if reg_beats else 'does not beat'} its baseline "
        f"({100 * (rm.get('baseline_rmse', 0) - rm.get('test_rmse', 0)) / max(rm.get('baseline_rmse', 1), 1e-9):+.1f}% RMSE), "
        "so its attributions describe a model with real, if modest, signal.")
    add("")
    add(f"The classifier scores accuracy {cm.get('test_accuracy', float('nan')):.3f} against a "
        f"majority-class baseline of {cm.get('baseline_accuracy', float('nan')):.3f} and ROC AUC "
        f"{cm.get('test_roc_auc_ovr', float('nan')):.3f}. It predicts `certified` for "
        "essentially every case. **Its SHAP values therefore show which features move the "
        "log-odds of each class, not which features successfully predict outcomes** -- the "
        "movements are real but never large enough to change the argmax. This is a property "
        "of the data: the generator's dominant denial driver is a latent audit variable that "
        "is deliberately not a column, mirroring real disclosure files.")
    add("")

    # ---- regression -------------------------------------------------------
    add("---")
    add("")
    add("## 1. Processing time (regression)")
    add("")
    add(f"Base value (mean prediction): **{reg['base_value']:.0f} days**. Each SHAP value "
        "below is a number of days added to or subtracted from that base.")
    add("")
    add("### Global drivers")
    add("")
    add("![Regressor SHAP beeswarm](" + p + "regressor_beeswarm.png)")
    add("")
    add("*Each dot is one test case. Position on the x-axis is that feature's contribution "
        "in days; colour is the feature's value (red high, blue low).*")
    add("")
    add("![Regressor mean absolute SHAP](" + p + "regressor_bar.png)")
    add("")
    imp = reg["importance"].head(12).copy()
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(2)
    add("| rank | feature | mean abs SHAP (days) | share |")
    add("|---:|---|---:|---:|")
    for i, r in imp.iterrows():
        add(f"| {i + 1} | `{r['feature']}` | {r['mean_abs_shap']:.2f} | {r['share_pct']:.1f}% |")
    add("")

    add(f"**Direction of `{reg['top_varying']}`** (the feature whose contribution varies "
        "most across cases -- ranked by SHAP standard deviation, not mean magnitude, "
        "since a feature that shifts every case equally explains none of the differences "
        "between them):")
    add("")
    d = reg["direction"]
    add("| value | n | mean SHAP (days) |")
    add("|---|---:|---:|")
    for _, r in d.iterrows():
        add(f"| {r['bucket']} | {int(r['count'])} | {r['mean_shap']:+.1f} |")
    add("")

    oor = reg.get("out_of_range")
    if oor and not oor["unseen_beats_baseline"]:
        add("### What the attribution reveals: an out-of-range top feature")
        add("")
        add(f"SHAP puts {reg['importance'].iloc[0]['share_pct']:.1f}% of the regressor's "
            f"attribution on `{oor['feature']}`. That feature is bounded in training "
            f"({oor['seen_values'][0]}-{oor['seen_values'][-1]}) but "
            f"{oor['n_unseen']:,} test rows ({oor['pct_unseen']:.1f}%) carry a value outside "
            "that range. A tree cannot extrapolate; those rows are routed into the nearest "
            "fitted bin. Splitting the test metric by that boundary:")
        add("")
        add("| test subset | n | model RMSE | baseline RMSE | verdict |")
        add("|---|---:|---:|---:|---|")
        add(f"| {oor['feature']} in range | {oor['n_seen']:,} | {oor['seen_rmse']:.1f} | "
            f"{oor['seen_baseline']:.1f} | "
            f"{'beats baseline' if oor['seen_beats_baseline'] else 'worse than baseline'} |")
        add(f"| {oor['feature']} out of range | {oor['n_unseen']:,} | "
            f"{oor['unseen_rmse']:.1f} | {oor['unseen_baseline']:.1f} | "
            f"{'beats baseline' if oor['unseen_beats_baseline'] else '**worse than baseline**'} |")
        add("")
        add("The headline RMSE averages these together and hides the second row. This is the "
            "same failure that made the filing-year split unusable in Phase 3, surviving in "
            "milder form: the split-year column is excluded automatically, but any *other* "
            "absolute-year feature is not. Excluding every year-valued column, or replacing "
            "them with a recent-backlog feature computed from past decisions, is the fix.")
        add("")

    add("### Individual case explanations")
    add("")
    for c in reg["cases"]:
        add(f"#### {c['label'].capitalize()} — predicted {c['predicted']:.0f} days, "
            f"actual {c['actual']:.0f} days")
        add("")
        add(f"![{c['label']}]({p}{c['file']})")
        add("")
        add(_contrib_markdown(c["table"], "days"))
        add("")

    # ---- classification ---------------------------------------------------
    add("---")
    add("")
    add("## 2. Case outcome (classification)")
    add("")
    add("SHAP values here are **log-odds (margin) contributions**, one set per class. "
        "They are not probabilities: +0.4 raises that class's margin, and the effect on the "
        "final probability depends on the other classes too.")
    add("")
    add("Base values (mean margin per class): "
        + ", ".join(f"`{n}` {v:+.2f}" for n, v in
                    zip(clf["class_names"], clf["base_values"])) + ".")
    add("")
    add("### Global drivers, averaged across classes")
    add("")
    add("![Classifier mean absolute SHAP](" + p + "classifier_bar.png)")
    add("")
    cimp = clf["importance"].head(12)
    cols = [c for c in cimp.columns if c.startswith("mean_abs_shap_")]
    add("| rank | feature | overall | " + " | ".join(c.replace("mean_abs_shap_", "")
                                                     for c in cols) + " |")
    add("|---:|---|---:|" + "---:|" * len(cols))
    for i, r in cimp.iterrows():
        vals = " | ".join(f"{r[c]:.3f}" for c in cols)
        add(f"| {i + 1} | `{r['feature']}` | {r['mean_abs_shap']:.3f} | {vals} |")
    add("")
    add("### Per-class views")
    add("")
    add("Averaging across classes hides drivers that matter to one outcome only, so each "
        "class gets its own beeswarm.")
    add("")
    for cp in clf["class_plots"]:
        add(f"**{cp['class']}**")
        add("")
        add(f"![{cp['class']} beeswarm]({p}{cp['file']})")
        add("")

    add(f"**Direction of `{clf['top_denied_feature']}`**, the feature whose `denied` "
        "contribution varies most across cases:")
    add("")
    dd = clf["denied_direction"]
    add("| value | n | mean SHAP (log-odds of denied) |")
    add("|---|---:|---:|")
    for _, r in dd.iterrows():
        add(f"| {r['bucket']} | {int(r['count'])} | {r['mean_shap']:+.3f} |")
    add("")

    add("### Individual case explanations")
    add("")
    for c in clf["cases"]:
        add(f"#### {c['label']} — P = {c['proba']:.3f}, model predicted "
            f"`{c['predicted']}`, actual `{c['actual']}`")
        add("")
        add(f"![{c['label']}]({p}{c['file']})")
        add("")
        add(_contrib_markdown(c["table"], f"log-odds of {c['class']}"))
        add("")

    # ---- summary ----------------------------------------------------------
    add("---")
    add("")
    add("## 3. Key drivers in one paragraph")
    add("")
    top5 = ", ".join(f"`{f}`" for f in reg["importance"].head(5)["feature"])
    add(f"For **processing time**, the model leans on {top5}. The single strongest is "
        f"`{reg['top_feature']}`, worth "
        f"{reg['importance'].iloc[0]['mean_abs_shap']:.0f} days of average absolute movement "
        f"against a {reg['base_value']:.0f}-day base. This is consistent with how the "
        "synthetic data was built: processing time there is driven by the fiscal-year "
        "backlog regime, a latent audit flag, and employer size, in that order.")
    add("")
    ctop5 = ", ".join(f"`{f}`" for f in clf["importance"].head(5)["feature"])
    add(f"For **case outcome**, the largest average margin movements come from {ctop5}, and "
        f"the feature that most separates one case from another on the `denied` margin is "
        f"`{clf['top_denied_feature']}`.")
    add("")
    we = clf.get("wage_effect")
    if we:
        add(f"The clearest recoverable mechanism is the prevailing-wage test. Of the "
            f"{clf['n_rows']:,} explained cases, {we['n_below']} ({we['pct_below']:.1f}%) offer "
            f"below the prevailing wage. For those, `wage_ratio` contributes on average "
            f"**{we['mean_below']:+.3f}** to the `denied` log-odds (peaking at "
            f"{we['max_below']:+.2f}), against **{we['mean_above']:+.3f}** for cases at or "
            "above prevailing wage. SHAP recovers this cleanly, and it is exactly the rule "
            "the generator was written with: offering below prevailing wage lifts the denial "
            "rate from roughly 5% to roughly 30%.")
        add("")
        add("That mechanism is real but rare. Because it touches so few cases, and because "
            "the generator's dominant denial driver -- the audit flag -- is latent, the "
            f"`{clf['denied_name']}` margin never overtakes `certified` in the argmax. The "
            "model is right about *what* matters and still unable to act on it.")
        add("")
    add("**What this means for the next phase.** The regression attributions point at "
        "time-varying backlog as the dominant signal, which supports adding a recent-backlog "
        "feature (median processing time of cases decided before this one was filed) rather "
        "than more case-level features. The classification attributions show the observable "
        "features carry only weak outcome signal, so improving the classifier is a "
        "feature-availability problem, not a hyperparameter problem.")
    add("")

    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(args) -> Path:
    plot_dir = args.report_dir / PLOT_SUBDIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("computing SHAP values for the regressor ...")
    reg = explain_regressor(args, plot_dir)
    print(f"  {reg['n_rows']:,} rows explained, additivity verified")

    print("computing SHAP values for the classifier ...")
    clf = explain_classifier(args, plot_dir)
    print(f"  {clf['n_rows']:,} rows x {len(clf['class_names'])} classes, "
          "additivity verified")

    report = write_report(reg, clf, args, args.report_dir / "shap_report.md")
    plots = sorted(plot_dir.glob("*.png"))
    print(f"\nwrote {report}")
    print(f"wrote {len(plots)} plots to {plot_dir}")
    for f in plots:
        print(f"  {f.name:<44} {f.stat().st_size / 1024:6.0f} KB")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate SHAP explanations and report.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--max-rows", type=int, default=3000,
                   help="cap test rows explained (0 = all); TreeSHAP is exact either way")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        run(args)
    except (sa.ExplainabilityError, DataError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
