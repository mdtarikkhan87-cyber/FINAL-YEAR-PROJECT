"""Model loading, feature construction, prediction, and per-request SHAP.

The feature row is built with the **same functions the training pipeline used**
-- the SOC crosswalk, the wage annualiser, the state-to-region map, the employer
size bands. Re-implementing any of them here would create training/serving skew:
the model would be fed a subtly different feature than the one it learned, and
nothing would fail loudly.

Column order and category levels come from the saved bundle rather than from
this module's own idea of them, so the row handed to XGBoost matches the fitted
model exactly. A category value the model never saw becomes NA, which is what
training did with unseen levels too.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.data_prep import cleaning as cl
from src.data_prep.features import EMPLOYER_SIZE_BINS, EMPLOYER_SIZE_LABELS
from src.data_prep.reference import STATE_TO_REGION, WAGE_UNIT_TO_ANNUAL
from src.explainability.shap_analysis import load_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the models are available."""


def _annualise(amount: float, unit: str) -> float:
    return float(amount) * WAGE_UNIT_TO_ANNUAL[unit.upper()]


def derive_features(case) -> dict[str, Any]:
    """Turn a CaseInput into the full derived feature dictionary.

    Mirrors ``src/data_prep/features.build_features`` field for field.
    """
    filing = pd.Series([pd.Timestamp(case.filing_date)])

    soc = cl.standardise_soc(pd.Series([case.soc_code]))
    soc, _ = cl.crosswalk_soc_2010_to_2018(soc)
    soc_code = soc.iloc[0]

    pw_annual = _annualise(case.prevailing_wage, case.prevailing_wage_unit)
    ow_annual = _annualise(case.offered_wage, case.offered_wage_unit)
    wage_ratio = ow_annual / pw_annual if pw_annual else np.nan

    size_bucket = None
    if case.employer_num_employees is not None:
        cut = pd.cut(pd.Series([float(case.employer_num_employees)]),
                     bins=EMPLOYER_SIZE_BINS, labels=EMPLOYER_SIZE_LABELS, right=False)
        size_bucket = None if pd.isna(cut.iloc[0]) else str(cut.iloc[0])

    employer_age = None
    if case.employer_year_established is not None:
        age = case.filing_date.year - case.employer_year_established
        employer_age = float(age) if 0 <= age <= 250 else None

    return {
        # employer
        "employer_size_bucket": size_bucket,
        "employer_num_employees": (float(case.employer_num_employees)
                                   if case.employer_num_employees is not None else np.nan),
        "employer_filing_volume": float(case.employer_filing_volume),
        "employer_age_years": employer_age if employer_age is not None else np.nan,
        "naics_sector": case.naics_code[:2] if case.naics_code else None,
        # occupation
        "soc_code": soc_code,
        "soc_major_group": cl.soc_major_group(pd.Series([soc_code])).iloc[0],
        "skill_level": case.skill_level.upper() if case.skill_level else None,
        # geography
        "worksite_state": case.worksite_state,
        "worksite_region": STATE_TO_REGION.get(case.worksite_state),
        # wages
        "prevailing_wage_annual": pw_annual,
        "offered_wage_annual": ow_annual,
        "wage_ratio": wage_ratio,
        "wage_premium": ow_annual - pw_annual,
        "offered_below_prevailing": float(wage_ratio < 1.0),
        # job requirements
        "job_education": case.job_education.upper() if case.job_education else None,
        "job_experience_months": (float(case.job_experience_months)
                                  if case.job_experience_months is not None else np.nan),
        "requires_experience": float(bool(case.job_experience_months or 0)),
        "job_alt_occupation": float(case.job_alt_occupation),
        "job_foreign_lang_req": float(case.job_foreign_lang_req),
        "job_combo_occupation": float(case.job_combo_occupation),
        "job_req_normal": float(case.job_req_normal),
        "layoff_past_six_months": float(case.layoff_past_six_months),
        "professional_occupation": float(case.professional_occupation),
        "refile": float(case.refile),
        # filing context
        "filing_fiscal_year": float(cl.fiscal_year(filing).iloc[0]),
        "filing_fiscal_quarter": float(cl.fiscal_quarter(filing).iloc[0]),
        "filing_month": float(case.filing_date.month),
        # representation and worker
        "has_attorney": float(case.has_attorney),
        "class_of_admission": (case.class_of_admission.strip().upper()
                               if case.class_of_admission else None),
        "citizenship": case.citizenship.strip().upper() if case.citizenship else None,
    }


class BundlePredictor:
    """One saved model plus everything needed to reproduce its feature row."""

    def __init__(self, bundle_path: Path, task: str) -> None:
        self.model, self.bundle = load_bundle(bundle_path)
        self.task = task
        self.features: list[str] = self.bundle["features"]
        self.categories: dict[str, list] = self.bundle.get("categories", {})
        self.classes: list[str] | None = self.bundle.get("classes")
        self.metrics: dict = self.bundle.get("metrics", {})
        self.split: dict = self.bundle.get("split", {})
        self._explainer = shap.TreeExplainer(self.model)

    def build_row(self, derived: dict[str, Any]) -> pd.DataFrame:
        """Assemble the single-row frame in the model's own column order."""
        row = {}
        for col in self.features:
            row[col] = derived.get(col, np.nan)
        frame = pd.DataFrame([row], columns=self.features)

        for col in self.features:
            if col in self.categories:
                dtype = pd.CategoricalDtype(categories=self.categories[col], ordered=False)
                # A level the model never saw becomes NA, exactly as in training.
                frame[col] = frame[col].astype("string").astype(dtype)
            else:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        return frame

    def unseen_categories(self, derived: dict[str, Any]) -> list[str]:
        """Which supplied categorical values the model has never encountered."""
        out = []
        for col, levels in self.categories.items():
            value = derived.get(col)
            if value is not None and not pd.isna(value) and str(value) not in levels:
                out.append(f"{col}={value}")
        return out

    def shap_for(self, frame: pd.DataFrame, class_index: int | None = None
                 ) -> tuple[float, np.ndarray]:
        """SHAP contributions for a single row: (base_value, values)."""
        explanation = self._explainer(frame)
        values = explanation.values
        base = explanation.base_values
        if class_index is None:
            return float(np.ravel(base)[0]), np.asarray(values)[0]
        return float(np.atleast_2d(base)[0][class_index]), np.asarray(values)[0, :, class_index]


class PredictorRegistry:
    """Both models, loaded once at startup."""

    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.regressor: BundlePredictor | None = None
        self.classifier: BundlePredictor | None = None
        self.load_errors: list[str] = []

    def load(self) -> None:
        for attr, filename, task in (
            ("regressor", "processing_time_xgb_bundle.joblib", "regression"),
            ("classifier", "outcome_xgb_bundle.joblib", "classification"),
        ):
            try:
                setattr(self, attr, BundlePredictor(self.artifact_dir / filename, task))
            except Exception as exc:  # noqa: BLE001 - surfaced through /health
                self.load_errors.append(f"{attr}: {exc}")

    @property
    def loaded(self) -> list[str]:
        return [n for n in ("regressor", "classifier") if getattr(self, n) is not None]

    def require(self, name: str) -> BundlePredictor:
        predictor = getattr(self, name, None)
        if predictor is None:
            raise ModelNotLoadedError(
                f"the {name} is not loaded. Train it first:\n"
                f"  python -m src.models.train_{'regressor' if name == 'regressor' else 'classifier'}"
            )
        return predictor


def contribution_records(features: list[str], frame: pd.DataFrame,
                         values: np.ndarray, top_n: int = 10) -> list[dict[str, Any]]:
    """Largest-magnitude contributions, with the value that produced each."""
    rows = []
    for name, shap_value in zip(features, values):
        raw = frame.iloc[0][name]
        display = "missing" if pd.isna(raw) else (
            f"{raw:,.2f}".rstrip("0").rstrip(".") if isinstance(raw, (int, float, np.floating))
            else str(raw)
        )
        rows.append({
            "feature": name,
            "value": display,
            "shap_value": float(shap_value),
            "direction": "increases" if shap_value >= 0 else "decreases",
        })
    rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
    return rows[:top_n]


def input_warnings(features: list[str], frame: pd.DataFrame, values: np.ndarray,
                   unseen: list[str], share_threshold: float = 0.10) -> list[str]:
    """Warn about omitted or unrecognised inputs that materially move the answer.

    Optional fields are not free. A field left out arrives at the model as NA and
    is routed down the missing branch -- and for a column that was never missing
    during training, that branch was never really fitted. The effect can be
    large and is invisible to the caller unless it is said out loud, so anything
    absorbing more than ``share_threshold`` of the total attribution is flagged
    with the size of its effect.
    """
    warnings: list[str] = []
    total = float(np.abs(values).sum())
    if total > 0:
        for name, shap_value in zip(features, values):
            if not pd.isna(frame.iloc[0][name]):
                continue
            share = abs(float(shap_value)) / total
            if share >= share_threshold:
                warnings.append(
                    f"'{name}' was not supplied and is moving this prediction by "
                    f"{float(shap_value):+.1f} ({share:.0%} of the total attribution). "
                    "Supply it for a more reliable answer."
                )
    for item in unseen:
        warnings.append(
            f"{item} is a value the model never saw in training; it is treated as "
            "missing and contributes nothing reliable."
        )
    return warnings


def estimated_decision_date(filing: date, days: float) -> date:
    return filing + timedelta(days=int(round(days)))
