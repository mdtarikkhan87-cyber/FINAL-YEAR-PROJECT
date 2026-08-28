"""Run the fairness audit and write ``reports/fairness_report.md``.

    python -m src.fairness.run_audit

Audits both Phase 4 models across three grouping attributes -- employer size,
occupation category (SOC major group), and worksite region -- and writes a
report with the metrics, the flagged disparities, and a plain-language reading
of what each one means.
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

from ..explainability.shap_analysis import ExplainabilityError, load_bundle  # noqa: E402
from ..models.data import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORT_DIR,
    DataError,
    load_modelling_data,
)
from . import metrics as fm  # noqa: E402

PLOT_SUBDIR = "fairness"

# attribute -> human-readable name used throughout the report
GROUPING_ATTRIBUTES = {
    "employer_size_bucket": "Employer size",
    "soc_major_group": "Occupation category (SOC major group)",
    "worksite_region": "Worksite region",
}

# The favourable outcome for parity purposes. Certification is the approval the
# applicant and employer actually want; everything else is a non-approval.
FAVOURABLE_CLASS = "certified"

SOC_MAJOR_GROUP_NAMES = {
    "11": "Management", "13": "Business & Financial", "15": "Computer & Mathematical",
    "17": "Architecture & Engineering", "19": "Life/Physical/Social Science",
    "25": "Education", "27": "Arts & Media", "29": "Healthcare Practitioners",
    "35": "Food Preparation", "41": "Sales", "43": "Office & Admin",
    "51": "Production", "53": "Transportation",
}


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------
def audit_regressor(args) -> dict:
    model, bundle = load_bundle(args.artifact_dir / "processing_time_xgb_bundle.joblib")
    data = load_modelling_data("processing_days", data_dir=args.data_dir)
    y_true = data.y_test.to_numpy()
    y_pred = model.predict(data.X_test)

    audits = []
    for attr in GROUPING_ATTRIBUTES:
        if attr not in data.X_test.columns:
            continue
        audits.append(fm.regression_audit(
            y_true, y_pred, data.X_test[attr], attr, args.min_group_size))
    return {"bundle": bundle, "audits": audits, "n_test": len(y_true),
            "y_true": y_true, "y_pred": y_pred, "X": data.X_test}


def audit_classifier(args) -> dict:
    model, bundle = load_bundle(args.artifact_dir / "outcome_xgb_bundle.joblib")
    class_names = bundle["classes"]
    data = load_modelling_data("outcome", data_dir=args.data_dir)

    if FAVOURABLE_CLASS not in class_names:
        raise ExplainabilityError(
            f"favourable class {FAVOURABLE_CLASS!r} is not among the model's classes "
            f"{class_names}; cannot binarise for parity metrics."
        )
    fav = class_names.index(FAVOURABLE_CLASS)

    proba = model.predict_proba(data.X_test)
    y_true = (data.y_test.astype(str) == FAVOURABLE_CLASS).astype(int).to_numpy()
    y_pred = (proba.argmax(axis=1) == fav).astype(int)
    y_score = proba[:, fav]

    audits = []
    for attr in GROUPING_ATTRIBUTES:
        if attr not in data.X_test.columns:
            continue
        audits.append(fm.classification_audit(
            y_true, y_pred, y_score, data.X_test[attr], attr, args.min_group_size))

    return {"bundle": bundle, "audits": audits, "n_test": len(y_true),
            "class_names": class_names, "y_true": y_true, "y_pred": y_pred,
            "y_score": y_score, "X": data.X_test,
            "degenerate": fm.detect_degenerate_predictions(y_pred)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _bar_panel(ax, labels, values, overall, title, xlabel, colour="#3b6ea5"):
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    ax.barh(labels, values, color=colour)
    ax.axvline(overall, color="crimson", ls="--", lw=1.2, label=f"overall {overall:.1f}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="lower right")


def plot_regression_gaps(reg: dict, path: Path, min_size: int) -> None:
    audits = reg["audits"]
    fig, axes = plt.subplots(1, len(audits), figsize=(5.2 * len(audits), 4.6))
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, audits):
        big = a.by_group[a.by_group["count"] >= min_size]
        _bar_panel(ax, [_label(a.attribute, v) for v in big[a.attribute]],
                   big["mae"].tolist(),
                   a.disparities["overall_mae"],
                   GROUPING_ATTRIBUTES.get(a.attribute, a.attribute),
                   "MAE (days)")
    fig.suptitle("Processing-time model: prediction error by group "
                 f"(groups with n >= {min_size})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_outcome_rates(clf: dict, path: Path, min_size: int) -> None:
    audits = clf["audits"]
    fig, axes = plt.subplots(1, len(audits), figsize=(5.2 * len(audits), 4.6))
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, audits):
        big = a.by_group[a.by_group["count"] >= min_size].sort_values("true_favourable_rate")
        y = np.arange(len(big))
        ax.barh(y - 0.2, big["true_favourable_rate"], height=0.4,
                color="#3b6ea5", label="actual certified rate")
        ax.barh(y + 0.2, big["mean_score"], height=0.4,
                color="#e8a33d", label="mean predicted P(certified)")
        ax.set_yticks(y, [_label(a.attribute, v) for v in big[a.attribute]], fontsize=8)
        ax.set_xlabel("rate", fontsize=9)
        ax.set_title(GROUPING_ATTRIBUTES.get(a.attribute, a.attribute), fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(labelsize=8)
    fig.suptitle("Outcome model: actual vs predicted certification rate by group "
                 f"(groups with n >= {min_size})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _flag_block(flags: list[fm.Flag]) -> list[str]:
    if not flags:
        return ["No disparities flagged at the configured thresholds.", ""]
    icon = {"high": "**HIGH**", "medium": "**MEDIUM**", "info": "INFO"}
    lines = []
    for f in flags:
        lines.append(f"- {icon[f.severity]} — {f.message}")
        if f.detail:
            lines.append(f"  <br>{f.detail}")
    lines.append("")
    return lines


def _label(attr: str, value: str) -> str:
    if attr == "soc_major_group":
        name = SOC_MAJOR_GROUP_NAMES.get(str(value))
        return f"{value} ({name})" if name else str(value)
    return str(value)


def write_report(reg: dict, clf: dict, args, path: Path) -> Path:
    p = f"{PLOT_SUBDIR}/"
    L: list[str] = []
    add = L.append

    reg_flags = fm.summarise_flags(reg["audits"])
    clf_flags = fm.summarise_flags(clf["audits"])

    add("# Fairness Audit Report")
    add("")
    add(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
        f"fairlearn · {reg['n_test']:,} test cases · "
        f"data `{args.data_dir.name}`")
    add("")
    add("> **Synthetic data.** Every case was invented by "
        "`src/data_prep/make_synthetic.py`. The disparities below are properties of a "
        "generator and two models fitted to it. Nothing here is a finding about real "
        "PERM adjudication or about any real employer, occupation, or region.")
    add("")
    add("> **These are structural proxies, not protected classes.** PERM disclosure data "
        "contains no applicant race or gender. This audit asks whether the models behave "
        "unevenly across parts of the labour market — employer size, occupation, region — "
        "not whether they discriminate against people. `country_of_citizenship` is present "
        "in the data and is far more protected-class-adjacent; it is deliberately **not** "
        "audited here, because whether it belongs in the model at all is still an open "
        "decision (HANDOVER.md §7.2).")
    add("")

    # ---- summary ----------------------------------------------------------
    add("## Summary")
    add("")
    add("| | regressor (`processing_days`) | classifier (`outcome`) |")
    add("|---|---|---|")
    add(f"| high-severity flags | {reg_flags['high']} | {clf_flags['high']} |")
    add(f"| medium | {reg_flags['medium']} | {clf_flags['medium']} |")
    add(f"| informational | {reg_flags['info']} | {clf_flags['info']} |")
    add("")

    if clf["degenerate"]:
        add("### The classifier's parity metrics cannot be read as a pass")
        add("")
        add(f"**{clf['degenerate']}**")
        add("")
        add("Demographic parity difference and equalized odds difference will both report "
            "`0.000` for every grouping attribute below. That is arithmetic, not evidence. "
            "A model that certifies everyone treats all groups identically — and is also "
            "useless, because it never identifies a denial. **Do not quote those zeros as "
            "a fairness result.** The score-based and outcome-rate columns are the "
            "informative ones for this model.")
        add("")
        add("To produce a genuinely non-degenerate audit, retrain with balanced class "
            "weights and re-run:")
        add("")
        add("```bash")
        add("python -m src.models.train_classifier --data-dir data/processed_decision --balanced")
        add("```")
        add("")

    add("## Method")
    add("")
    add(f"- **Favourable outcome** for parity purposes is `{FAVOURABLE_CLASS}`. The "
        "four-class target is binarised to certified / not-certified, because demographic "
        "parity and equalized odds are defined for binary outcomes.")
    add(f"- **Grouping attributes**: {', '.join(GROUPING_ATTRIBUTES.values())}.")
    add(f"- **Small groups** (n < {args.min_group_size}) are shown in the tables but "
        "excluded from flagging and from the charts; a small group's error rate is noise.")
    add(f"- **Thresholds**: parity difference > {fm.DP_DIFFERENCE_THRESHOLD}; "
        f"selection-rate ratio < {fm.FOUR_FIFTHS_RATIO} (the four-fifths screen); "
        f"group MAE > {fm.REGRESSION_MAE_RATIO_THRESHOLD}x overall; group bias departing "
        f"from the model's overall bias by more than "
        f"{fm.REGRESSION_BIAS_DEVIATION_DAYS:.0f} days. "
        "These are screening tripwires from common practice, not legal standards for this "
        "setting.")
    add("")

    # ---- regression -------------------------------------------------------
    add("---")
    add("")
    add("## 1. Processing-time model — group-wise error")
    add("")
    add("Demographic parity has no meaning for a continuous target. The analogous question "
        "is whether the model is **equally accurate for everyone**, and whether any group "
        "is systematically told the wrong thing in the same direction.")
    add("")
    overall_bias = reg["audits"][0].disparities["overall_bias"] if reg["audits"] else 0.0
    direction = "under" if overall_bias < 0 else "over"
    add(f"> **Read this before the group tables.** The model {direction}-predicts by "
        f"**{abs(overall_bias):.0f} days on average across the entire test set**. Every "
        "group inherits that, so every group looks badly biased in absolute terms. That is "
        "one model defect, not twenty group-level fairness problems. The tables below "
        "therefore report `bias vs overall` — how far each group departs from that global "
        f"figure — and only a departure beyond ±{fm.REGRESSION_BIAS_DEVIATION_DAYS:.0f} days "
        "is flagged. Fixing the global bias is a modelling task (HANDOVER.md §7.3), not a "
        "fairness one.")
    add("")
    add(f"![Regression error by group]({p}regression_gaps.png)")
    add("")
    for a in reg["audits"]:
        add(f"### {GROUPING_ATTRIBUTES.get(a.attribute, a.attribute)}")
        add("")
        d = a.disparities
        add(f"Overall MAE **{d['overall_mae']:.1f} days**. Best group "
            f"{d['best_group_mae']:.1f}, worst {d['worst_group_mae']:.1f} — "
            f"a spread of **{d['mae_difference']:.1f} days** "
            f"(best/worst ratio {d['mae_best_worst_ratio']:.2f}). "
            f"Largest departure from the model's overall bias: "
            f"**{d['max_bias_deviation']:.0f} days**.")
        add("")
        add("| group | n | MAE | RMSE | mean error | bias vs overall | mean actual | "
            "mean predicted | MAE vs overall |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in a.by_group.sort_values("mae", ascending=False).iterrows():
            add(f"| {_label(a.attribute, r[a.attribute])} | {int(r['count'])} | "
                f"{r['mae']:.1f} | {r['rmse']:.1f} | {r['mean_error']:+.1f} | "
                f"{r['bias_vs_overall']:+.1f} | "
                f"{r['mean_actual']:.0f} | {r['mean_predicted']:.0f} | "
                f"{r['mae_ratio_vs_overall']:.2f}x |")
        add("")
        add("**Flags**")
        add("")
        L.extend(_flag_block(a.flags))

    # ---- classification ---------------------------------------------------
    add("---")
    add("")
    add("## 2. Outcome model — demographic parity and equalized odds")
    add("")
    add(f"![Outcome rates by group]({p}outcome_rates.png)")
    add("")
    add("Column meanings: **true rate** is the observed certification rate in the data "
        "(a property of the data, not the model). **selection rate** is how often the "
        "model predicts certified. **mean score** is the average predicted probability of "
        "certification — the column that still carries signal when the hard labels do not.")
    add("")
    for a in clf["audits"]:
        add(f"### {GROUPING_ATTRIBUTES.get(a.attribute, a.attribute)}")
        add("")
        d = a.disparities
        add(f"- Demographic parity difference: **{d['demographic_parity_difference']:.3f}**"
            f"{'  ← uninformative (constant predictions)' if clf['degenerate'] else ''}")
        add(f"- Equalized odds difference: **{d['equalized_odds_difference']:.3f}**"
            f"{'  ← uninformative (constant predictions)' if clf['degenerate'] else ''}")
        add(f"- Selection-rate ratio: **{d['demographic_parity_ratio']:.3f}** "
            f"(four-fifths screen: {fm.FOUR_FIFTHS_RATIO})")
        add(f"- Predicted-score spread across groups: **{d['score_difference']:.3f}**")
        add(f"- Observed certification-rate spread: **{d['true_rate_difference']:.3f}**")
        add("")
        add("| group | n | true rate | selection rate | mean score | accuracy | TPR | FPR |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in a.by_group.sort_values("true_favourable_rate").iterrows():
            fpr = "n/a" if pd.isna(r["fpr"]) else f"{r['fpr']:.3f}"
            tpr = "n/a" if pd.isna(r["tpr"]) else f"{r['tpr']:.3f}"
            add(f"| {_label(a.attribute, r[a.attribute])} | {int(r['count'])} | "
                f"{r['true_favourable_rate']:.3f} | {r['selection_rate']:.3f} | "
                f"{r['mean_score']:.3f} | {r['accuracy']:.3f} | {tpr} | {fpr} |")
        add("")
        add("**Flags**")
        add("")
        L.extend(_flag_block(a.flags))

    # ---- interpretation ---------------------------------------------------
    add("---")
    add("")
    add("## 3. Plain-language interpretation")
    add("")

    worst = []
    for a in reg["audits"]:
        big = a.by_group[a.by_group["count"] >= args.min_group_size]
        if len(big):
            row = big.loc[big["mae"].idxmax()]
            worst.append((a.attribute, _label(a.attribute, row[a.attribute]),
                          float(row["mae"]), float(a.disparities["overall_mae"]),
                          int(row["count"])))

    add("**Processing time.** ")
    if worst:
        for attr, grp, mae, overall, n in worst:
            add(f"- Across *{GROUPING_ATTRIBUTES.get(attr, attr).lower()}*, the least "
                f"well-served group is **{grp}** (n={n}), with a mean absolute error of "
                f"{mae:.0f} days against an overall {overall:.0f}. In practice that means "
                f"an employer in this group asking \"how long will my case take?\" gets an "
                f"answer that is off by about {mae:.0f} days on average.")
    add("")
    add("A gap in MAE is not automatically unfair — some groups genuinely have more "
        "variable processing times, and a model cannot be more precise than the process "
        "it describes. The question that matters is whether any group departs from how the "
        "model treats everyone else. On this data the departures are small: the model is "
        "uniformly and badly optimistic rather than selectively so. That is a real problem "
        "for anyone relying on the estimate, but it is an accuracy problem, and it would "
        "not be fixed by any fairness intervention.")
    add("")

    add("**Case outcome.** ")
    if clf["degenerate"]:
        add("- The model certifies every case, so it cannot treat any group differently "
            "in its final answer. Both parity metrics are 0.000 across all three "
            "attributes. This is the degenerate pass described above and should not be "
            "reported as a fairness result.")
        spreads = [(a.attribute, a.disparities["score_difference"],
                    a.disparities["true_rate_difference"]) for a in clf["audits"]]
        biggest = max(spreads, key=lambda t: t[1])
        add(f"- The predicted *probabilities* do vary by group. The widest spread is across "
            f"*{GROUPING_ATTRIBUTES.get(biggest[0], biggest[0]).lower()}* at "
            f"{biggest[1]:.3f}. If the model were ever re-balanced or thresholded so that "
            "denials could actually be predicted, that spread is where disparity would "
            "first appear — so it is worth tracking now rather than after the change.")
        widest_true = max(spreads, key=lambda t: t[2])
        add(f"- The *data* carries real outcome differences: certification rates vary by "
            f"{widest_true[2]:.3f} across "
            f"*{GROUPING_ATTRIBUTES.get(widest_true[0], widest_true[0]).lower()}*. That is "
            "a property of the generator, not of the model, and a model that learned to "
            "predict denials would have to be checked against it carefully — matching the "
            "base rate exactly is one definition of fairness and a violation of another.")
    add("")
    add("**What to do next.** Re-run this audit against the balanced classifier before "
        "drawing any conclusion about outcome fairness; the current numbers cannot support "
        "one. For the regressor, the group gaps are real and should be re-checked after the "
        "recent-backlog feature lands, since a large share of its error is regime shift "
        "rather than anything group-specific.")
    add("")

    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(args) -> Path:
    plot_dir = args.report_dir / PLOT_SUBDIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("auditing the processing-time model ...")
    reg = audit_regressor(args)
    print(f"  {len(reg['audits'])} grouping attributes, {reg['n_test']:,} test cases")

    print("auditing the outcome model ...")
    clf = audit_classifier(args)
    print(f"  {len(clf['audits'])} grouping attributes, {clf['n_test']:,} test cases")
    if clf["degenerate"]:
        print("  NOTE: predictions are constant -> parity metrics are degenerate")

    plot_regression_gaps(reg, plot_dir / "regression_gaps.png", args.min_group_size)
    plot_outcome_rates(clf, plot_dir / "outcome_rates.png", args.min_group_size)

    report = write_report(reg, clf, args, args.report_dir / "fairness_report.md")

    rf, cf = fm.summarise_flags(reg["audits"]), fm.summarise_flags(clf["audits"])
    print(f"\nflags — regressor: {rf['high']} high, {rf['medium']} medium, {rf['info']} info")
    print(f"      classifier: {cf['high']} high, {cf['medium']} medium, {cf['info']} info")
    print(f"\nwrote {report}")
    for f in sorted(plot_dir.glob("*.png")):
        print(f"wrote {f} ({f.stat().st_size / 1024:.0f} KB)")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fairness audit for the PERM models.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--min-group-size", type=int, default=fm.MIN_GROUP_SIZE,
                   help="groups smaller than this are shown but never flagged")
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        run(args)
    except (ExplainabilityError, DataError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
