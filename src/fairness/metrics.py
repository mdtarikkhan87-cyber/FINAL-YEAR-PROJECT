"""Group fairness metrics for the PERM models.

Built on ``fairlearn.metrics``. Three things here are worth knowing before
reading any number this module produces.

**1. A constant classifier scores as perfectly fair.** Demographic parity and
equalized odds compare *hard predictions* across groups. A model that predicts
the same class for every case has identical selection rates and identical
TPR/FPR everywhere, so both differences come out at exactly 0.0. That is not
fairness, it is degeneracy, and it is the single easiest way to publish a
misleading fairness result. ``detect_degenerate_predictions`` catches it and the
report refuses to call it a pass.

**2. Score-based parity is reported alongside label-based parity.** Even when
the argmax never moves, the underlying probabilities do vary by group. Those
differences are what a threshold change or a re-balanced model would expose, so
they are audited whether or not the labels are degenerate.

**3. Base rates are audited separately from the model.** A gap in the *observed*
outcome rate across groups is a property of the data, not evidence that the
model discriminates. Both are reported, and the report keeps them apart.

The grouping attributes are structural proxies -- employer size, occupation,
region -- not protected classes. The disclosure data contains no applicant race
or gender. Findings should be phrased accordingly: this measures whether the
model behaves unevenly across parts of the labour market, not whether it
discriminates against people.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, mean_absolute_error

MISSING_LABEL = "<missing>"

# Thresholds. The 0.10 difference cut-off is the common working convention in
# the fairness literature; 0.80 on a ratio is the US "four-fifths rule" used by
# the EEOC as a screen for adverse impact. Neither is a legal standard for this
# setting -- they are screening tripwires, and the report says so.
DP_DIFFERENCE_THRESHOLD = 0.10
EO_DIFFERENCE_THRESHOLD = 0.10
FOUR_FIFTHS_RATIO = 0.80
REGRESSION_MAE_RATIO_THRESHOLD = 1.25
MIN_GROUP_SIZE = 30

# Directional bias is flagged on how far a group departs from the model's
# *overall* bias, not on its absolute value. The processing-time model
# under-predicts by ~89 days for everyone; repeating that global defect once per
# group would bury any real group-specific effect under 20+ identical flags.
# A group only matters here if it is treated differently from the rest.
REGRESSION_BIAS_DEVIATION_DAYS = 30.0


@dataclass
class Flag:
    severity: str          # "high" | "medium" | "info"
    attribute: str
    message: str
    detail: str = ""


@dataclass
class GroupAudit:
    attribute: str
    by_group: pd.DataFrame
    overall: dict
    disparities: dict
    flags: list[Flag] = field(default_factory=list)
    small_groups: list[str] = field(default_factory=list)


def prepare_groups(series: pd.Series, min_size: int = MIN_GROUP_SIZE
                   ) -> tuple[pd.Series, list[str]]:
    """Stringify a grouping column and report which levels are too small to read.

    Small groups are kept in the tables -- hiding them would be its own kind of
    dishonesty -- but flagged, because a 12-case group's MAE is noise and should
    never be reported as a disparity.
    """
    groups = series.astype("string").fillna(MISSING_LABEL)
    counts = groups.value_counts()
    small = sorted(counts[counts < min_size].index.tolist())
    return groups, small


def detect_degenerate_predictions(y_pred: np.ndarray) -> str | None:
    """Return an explanation if predictions are constant, else None."""
    unique = pd.unique(pd.Series(y_pred))
    if len(unique) <= 1:
        # np.int64(1) reads badly in a report; render the bare value.
        value = unique[0].item() if len(unique) and hasattr(unique[0], "item") else (
            unique[0] if len(unique) else "none")
        return (
            f"The model predicts the same value ({value}) for every test case, so every "
            "group has an identical selection rate and identical TPR/FPR. Demographic "
            "parity and equalized odds are therefore 0.0 by construction. This is a "
            "degenerate model, not a fair one -- the parity metrics carry no information "
            "here."
        )
    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    groups: pd.Series,
    attribute: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> GroupAudit:
    """Group metrics for a binarised outcome (1 = favourable).

    ``y_score`` is the predicted probability of the favourable class, audited in
    its own right so the result stays informative when the hard labels do not
    vary.
    """
    groups, small = prepare_groups(groups, min_group_size)

    frame = MetricFrame(
        metrics={
            "count": lambda yt, yp: len(yt),
            "true_favourable_rate": lambda yt, yp: float(np.mean(yt)),
            "selection_rate": selection_rate,
            "accuracy": accuracy_score,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y_true, y_pred=y_pred, sensitive_features=groups,
    )
    by_group = frame.by_group.copy()

    # Mean predicted score per group -- informative even when labels are constant.
    score_by_group = pd.Series(y_score).groupby(groups.reset_index(drop=True)).mean()
    by_group["mean_score"] = score_by_group
    by_group = by_group.reset_index().rename(columns={by_group.index.name or "index": attribute})
    by_group.columns = [attribute] + list(by_group.columns[1:])

    disparities = {
        "demographic_parity_difference": float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=groups)),
        "demographic_parity_ratio": float(
            demographic_parity_ratio(y_true, y_pred, sensitive_features=groups)),
        "equalized_odds_difference": float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=groups)),
        "accuracy_difference": float(frame.difference()["accuracy"]),
        "score_difference": float(score_by_group.max() - score_by_group.min()),
        "true_rate_difference": float(
            by_group["true_favourable_rate"].max() - by_group["true_favourable_rate"].min()),
    }

    flags = _flag_classification(attribute, by_group, disparities, y_pred,
                                 small, min_group_size)
    return GroupAudit(attribute=attribute, by_group=by_group,
                      overall={k: float(v) for k, v in frame.overall.items()},
                      disparities=disparities, flags=flags, small_groups=small)


def _flag_classification(attribute, by_group, disparities, y_pred, small,
                         min_group_size) -> list[Flag]:
    flags: list[Flag] = []

    degenerate = detect_degenerate_predictions(y_pred)
    if degenerate:
        flags.append(Flag(
            "high", attribute,
            "parity metrics are uninformative: predictions are constant",
            degenerate,
        ))
    else:
        if disparities["demographic_parity_difference"] > DP_DIFFERENCE_THRESHOLD:
            flags.append(Flag(
                "high", attribute,
                f"demographic parity difference "
                f"{disparities['demographic_parity_difference']:.3f} exceeds "
                f"{DP_DIFFERENCE_THRESHOLD}",
                "Groups receive the favourable prediction at materially different rates.",
            ))
        if disparities["demographic_parity_ratio"] < FOUR_FIFTHS_RATIO:
            flags.append(Flag(
                "high", attribute,
                f"selection-rate ratio {disparities['demographic_parity_ratio']:.3f} "
                f"falls below the four-fifths screen ({FOUR_FIFTHS_RATIO})",
                "The least-selected group is selected at under 80% of the rate of the most-selected.",
            ))
        if disparities["equalized_odds_difference"] > EO_DIFFERENCE_THRESHOLD:
            flags.append(Flag(
                "high", attribute,
                f"equalized odds difference "
                f"{disparities['equalized_odds_difference']:.3f} exceeds "
                f"{EO_DIFFERENCE_THRESHOLD}",
                "Error rates (TPR and/or FPR) differ materially across groups.",
            ))

    if disparities["score_difference"] > DP_DIFFERENCE_THRESHOLD:
        readable = by_group.sort_values("mean_score")
        flags.append(Flag(
            "medium", attribute,
            f"predicted-score spread {disparities['score_difference']:.3f} across groups",
            f"Lowest mean predicted probability: {readable.iloc[0][attribute]} "
            f"({readable.iloc[0]['mean_score']:.3f}); highest: "
            f"{readable.iloc[-1][attribute]} ({readable.iloc[-1]['mean_score']:.3f}). "
            "Hard labels may hide this, but a threshold change would expose it.",
        ))

    if disparities["true_rate_difference"] > DP_DIFFERENCE_THRESHOLD:
        readable = by_group.sort_values("true_favourable_rate")
        flags.append(Flag(
            "info", attribute,
            f"observed outcome rates differ by "
            f"{disparities['true_rate_difference']:.3f} across groups",
            f"This is a property of the data, not the model: {readable.iloc[0][attribute]} "
            f"has a {readable.iloc[0]['true_favourable_rate']:.3f} favourable rate versus "
            f"{readable.iloc[-1]['true_favourable_rate']:.3f} for "
            f"{readable.iloc[-1][attribute]}.",
        ))

    if small:
        flags.append(Flag(
            "info", attribute,
            f"{len(small)} group(s) below {min_group_size} cases",
            f"Metrics for {', '.join(small[:8])}"
            f"{' ...' if len(small) > 8 else ''} are too noisy to interpret.",
        ))
    return flags


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def regression_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    attribute: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> GroupAudit:
    """Group-wise error metrics for the processing-time model.

    Demographic parity has no meaning for a continuous target, so the analogous
    question is whether the model is *equally accurate* for everyone. A group
    whose predictions are systematically late or early is being underserved even
    if the aggregate error looks acceptable.
    """
    groups, small = prepare_groups(groups, min_group_size)

    frame = MetricFrame(
        metrics={
            "count": lambda yt, yp: len(yt),
            "mae": mean_absolute_error,
            "rmse": lambda yt, yp: float(np.sqrt(np.mean((np.asarray(yp) - np.asarray(yt)) ** 2))),
            "mean_error": lambda yt, yp: float(np.mean(np.asarray(yp) - np.asarray(yt))),
            "mean_actual": lambda yt, yp: float(np.mean(yt)),
            "mean_predicted": lambda yt, yp: float(np.mean(yp)),
        },
        y_true=y_true, y_pred=y_pred, sensitive_features=groups,
    )
    by_group = frame.by_group.copy().reset_index()
    by_group.columns = [attribute] + list(by_group.columns[1:])

    overall_mae = float(mean_absolute_error(y_true, y_pred))
    overall_bias = float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))
    by_group["mae_ratio_vs_overall"] = (by_group["mae"] / overall_mae).round(3)
    by_group["bias_vs_overall"] = (by_group["mean_error"] - overall_bias).round(1)

    disparities = {
        "overall_mae": overall_mae,
        "overall_bias": overall_bias,
        "mae_difference": float(frame.difference()["mae"]),
        "mae_best_worst_ratio": float(frame.ratio()["mae"]),
        "worst_group_mae": float(by_group["mae"].max()),
        "best_group_mae": float(by_group["mae"].min()),
        "bias_difference": float(by_group["mean_error"].max() - by_group["mean_error"].min()),
        "max_bias_deviation": float(by_group["bias_vs_overall"].abs().max()),
    }

    flags = _flag_regression(attribute, by_group, disparities, overall_mae,
                             small, min_group_size)
    return GroupAudit(attribute=attribute, by_group=by_group,
                      overall={k: float(v) for k, v in frame.overall.items()},
                      disparities=disparities, flags=flags, small_groups=small)


def _flag_regression(attribute, by_group, disparities, overall_mae, small,
                     min_group_size) -> list[Flag]:
    flags: list[Flag] = []
    big = by_group[by_group["count"] >= min_group_size]

    worse = big[big["mae_ratio_vs_overall"] > REGRESSION_MAE_RATIO_THRESHOLD]
    for _, row in worse.sort_values("mae_ratio_vs_overall", ascending=False).iterrows():
        flags.append(Flag(
            "high", attribute,
            f"group '{row[attribute]}' has MAE {row['mae']:.1f} days, "
            f"{row['mae_ratio_vs_overall']:.2f}x the overall {overall_mae:.1f}",
            f"n={int(row['count'])}. Predictions for this group are materially less "
            f"accurate than average.",
        ))

    biased = big[big["bias_vs_overall"].abs() > REGRESSION_BIAS_DEVIATION_DAYS]
    for _, row in biased.sort_values("bias_vs_overall", key=abs, ascending=False).iterrows():
        worse = "further under" if row["bias_vs_overall"] < 0 else "less under"
        flags.append(Flag(
            "medium", attribute,
            f"group '{row[attribute]}' departs from the model's overall bias by "
            f"{row['bias_vs_overall']:+.0f} days",
            f"n={int(row['count'])}. The model under-predicts everyone by "
            f"{abs(disparities['overall_bias']):.0f} days on average; this group is "
            f"{worse}-predicted still, at {row['mean_error']:+.0f} days "
            f"(mean actual {row['mean_actual']:.0f} vs predicted {row['mean_predicted']:.0f}). "
            "A directional error specific to one group is what makes it a fairness "
            "concern rather than a general accuracy problem.",
        ))

    if small:
        flags.append(Flag(
            "info", attribute,
            f"{len(small)} group(s) below {min_group_size} cases",
            f"Metrics for {', '.join(small[:8])}"
            f"{' ...' if len(small) > 8 else ''} are too noisy to interpret.",
        ))
    return flags


def summarise_flags(audits: list[GroupAudit]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "info": 0}
    for a in audits:
        for f in a.flags:
            counts[f.severity] = counts.get(f.severity, 0) + 1
    return counts
