"""SHAP computation and case selection for the PERM models.

Uses ``shap.TreeExplainer``, which for XGBoost runs exact TreeSHAP -- not a
sampling approximation -- and correctly respects ``best_iteration``, so the
explanations describe the same tree subset that ``predict``/``predict_proba``
actually use. Additivity is asserted rather than assumed: if the SHAP values
plus the base value do not reconstruct the model's own output, the run fails
instead of quietly producing a plausible-looking but wrong picture.

Shapes, since they differ by task and are easy to get wrong:

* regression      ``(n_rows, n_features)``          -- units are days
* classification  ``(n_rows, n_features, n_class)`` -- units are log-odds
  (margin) contributions, one slice per class. They are **not** probabilities;
  a contribution of +0.4 moves that class's margin, and the effect on the final
  probability depends on every other class's margin too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


class ExplainabilityError(RuntimeError):
    """Raised when a model bundle cannot be explained."""


@dataclass
class ShapResult:
    explanation: shap.Explanation
    X: pd.DataFrame
    feature_names: list[str]
    task: str
    class_names: list[str] | None = None
    base_values: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        return int(self.explanation.values.shape[0])

    def for_class(self, class_index: int) -> shap.Explanation:
        """2-D Explanation slice for one class of a multiclass model."""
        if self.task != "classification":
            return self.explanation
        return self.explanation[:, :, class_index]


def load_bundle(path: Path) -> tuple[object, dict]:
    """Load a saved model bundle written by ``src.models.tracking``."""
    if not path.exists():
        raise ExplainabilityError(
            f"model bundle not found: {path}\n"
            "Train the models first:\n"
            "    python -m src.models.train_regressor\n"
            "    python -m src.models.train_classifier"
        )
    bundle = joblib.load(path)
    model = bundle.get("model")
    if model is None:
        raise ExplainabilityError(f"{path} contains no 'model' key")
    return model, bundle


def compute_shap(model, X: pd.DataFrame, task: str,
                 class_names: list[str] | None = None,
                 check_additivity: bool = True) -> ShapResult:
    """Exact TreeSHAP over ``X``, with an additivity check against the model."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    if check_additivity:
        _assert_additive(model, X, explanation, task)

    return ShapResult(
        explanation=explanation, X=X, feature_names=list(X.columns),
        task=task, class_names=class_names,
        base_values=np.atleast_1d(np.asarray(explanation.base_values)[0]
                                  if explanation.base_values.ndim > 1
                                  else explanation.base_values[:1]),
    )


def _assert_additive(model, X: pd.DataFrame, explanation: shap.Explanation,
                     task: str, tol: float = 1e-2) -> None:
    """SHAP values + base value must reproduce the model's own output."""
    values = explanation.values
    base = explanation.base_values

    if task == "regression":
        recon = values.sum(axis=1) + np.ravel(base)
        actual = model.predict(X)
    else:
        recon = values.sum(axis=1) + np.atleast_2d(base)
        best = int(getattr(model, "best_iteration", 0) or 0)
        actual = model.get_booster().inplace_predict(
            X, predict_type="margin", iteration_range=(0, best + 1)
        )

    diff = float(np.abs(np.asarray(recon) - np.asarray(actual)).max())
    if diff > tol:
        raise ExplainabilityError(
            f"SHAP additivity check failed for the {task} model: reconstructed "
            f"output differs from the model's own by up to {diff:.4f} (tolerance "
            f"{tol}). The explanations would not describe this model -- most "
            "likely the explainer is using a different tree subset than predict()."
        )


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------
def global_importance(result: ShapResult) -> pd.DataFrame:
    """Mean |SHAP| per feature.

    For a classifier the per-class columns are kept alongside the overall mean,
    because a feature can matter a great deal to one class and not at all to the
    rest -- averaging that away is how a driver of denials gets hidden behind
    the dominant certified class.
    """
    values = result.explanation.values
    if result.task == "regression":
        out = pd.DataFrame({"feature": result.feature_names,
                            "mean_abs_shap": np.abs(values).mean(axis=0),
                            "shap_std": values.std(axis=0)})
    else:
        per_class = np.abs(values).mean(axis=0)  # (n_features, n_classes)
        out = pd.DataFrame({"feature": result.feature_names,
                            "mean_abs_shap": per_class.mean(axis=1),
                            "shap_std": values.std(axis=0).mean(axis=1)})
        for i, name in enumerate(result.class_names or []):
            out[f"mean_abs_shap_{name}"] = per_class[:, i]
            out[f"shap_std_{name}"] = values[:, :, i].std(axis=0)

    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = out["mean_abs_shap"].sum()
    out["share_pct"] = (100 * out["mean_abs_shap"] / total).round(2) if total else 0.0
    return out


def most_discriminating(importance: pd.DataFrame, class_name: str | None = None) -> str:
    """The feature whose contribution *varies* most across cases.

    Mean |SHAP| alone is a poor guide to what drives a prediction: a feature that
    shifts every case by the same amount scores highly while explaining none of
    the differences between them. Ranking by the standard deviation of the SHAP
    values picks the feature that actually separates cases.
    """
    col = f"shap_std_{class_name}" if class_name and f"shap_std_{class_name}" in importance \
        else "shap_std"
    return str(importance.sort_values(col, ascending=False).iloc[0]["feature"])


def directional_effect(result: ShapResult, feature: str,
                       class_index: int | None = None) -> pd.DataFrame:
    """Mean signed SHAP grouped by the feature's own value.

    Answers "which values of this feature push the prediction up, and by how
    much" -- the part a bare importance ranking leaves out.
    """
    idx = result.feature_names.index(feature)
    values = result.explanation.values
    contrib = values[:, idx] if class_index is None else values[:, idx, class_index]
    series = result.X[feature]
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 12:
        bucket = pd.qcut(series, q=5, duplicates="drop")
    else:
        bucket = series.astype("string").fillna("<missing>")

    frame = pd.DataFrame({"bucket": bucket.astype("string"), "shap": contrib})
    out = (frame.groupby("bucket", observed=True)["shap"]
           .agg(["count", "mean"]).reset_index()
           .rename(columns={"mean": "mean_shap"}))
    return out.sort_values("mean_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------
def select_regression_cases(predictions: np.ndarray, actuals: pd.Series,
                            n: int = 3) -> dict[str, int]:
    """Pick cases that span the prediction range rather than three near-identical ones."""
    order = np.argsort(predictions)
    picks = {
        "fastest predicted": int(order[0]),
        "typical (median prediction)": int(order[len(order) // 2]),
        "slowest predicted": int(order[-1]),
    }
    if n > 3:
        picks["largest under-prediction"] = int(np.argmax(actuals.to_numpy() - predictions))
    return picks


def select_classification_cases(proba: np.ndarray,
                                class_names: list[str]) -> dict[str, tuple[int, int]]:
    """Pick the most extreme case for each interesting class.

    Returns ``label -> (row_index, class_index)`` so each waterfall explains the
    class that case is notable for, not always the majority class.
    """
    picks: dict[str, tuple[int, int]] = {}
    for name in ("denied", "withdrawn", "certified"):
        if name not in class_names:
            continue
        ci = class_names.index(name)
        picks[f"highest P({name})"] = (int(np.argmax(proba[:, ci])), ci)
    return picks


def out_of_range_diagnostic(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_test: pd.Series, predictions: np.ndarray,
                            baseline_value: float, feature: str) -> dict | None:
    """Split test performance by whether ``feature``'s value was seen in training.

    A tree cannot extrapolate: a test value outside the fitted range is routed
    into the nearest existing bin. When SHAP shows the model leaning hard on a
    monotonically increasing feature such as a year, this checks whether the
    rows carrying an unseen value are quietly being predicted badly -- a failure
    an aggregate metric averages away.
    """
    if feature not in X_train.columns or feature not in X_test.columns:
        return None
    seen = set(pd.unique(X_train[feature].dropna()))
    unseen_mask = ~X_test[feature].isin(seen)
    if not unseen_mask.any() or unseen_mask.all():
        return None

    def _rmse(mask: np.ndarray) -> tuple[float, float, int]:
        actual = y_test.to_numpy()[mask]
        model_rmse = float(np.sqrt(np.mean((predictions[mask] - actual) ** 2)))
        base_rmse = float(np.sqrt(np.mean((baseline_value - actual) ** 2)))
        return model_rmse, base_rmse, int(mask.sum())

    m = unseen_mask.to_numpy()
    unseen_rmse, unseen_base, n_unseen = _rmse(m)
    seen_rmse, seen_base, n_seen = _rmse(~m)
    return {
        "feature": feature,
        "seen_values": sorted(int(v) for v in seen if pd.notna(v)),
        "n_unseen": n_unseen, "pct_unseen": 100.0 * n_unseen / len(X_test),
        "unseen_rmse": unseen_rmse, "unseen_baseline": unseen_base,
        "n_seen": n_seen, "seen_rmse": seen_rmse, "seen_baseline": seen_base,
        "unseen_beats_baseline": unseen_rmse < unseen_base,
        "seen_beats_baseline": seen_rmse < seen_base,
    }


def contribution_table(result: ShapResult, row: int, class_index: int | None = None,
                       top_n: int = 8) -> pd.DataFrame:
    """The features that moved one prediction most, with their actual values."""
    values = result.explanation.values
    contrib = values[row, :] if class_index is None else values[row, :, class_index]
    frame = pd.DataFrame({
        "feature": result.feature_names,
        "value": [result.X.iloc[row][f] for f in result.feature_names],
        "shap": contrib,
    })
    frame["abs"] = frame["shap"].abs()
    frame = frame.sort_values("abs", ascending=False).head(top_n)
    frame["direction"] = np.where(frame["shap"] >= 0, "increases", "decreases")
    return frame.drop(columns="abs").reset_index(drop=True)
