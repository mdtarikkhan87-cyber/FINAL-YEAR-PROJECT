"""Request and response schemas for the PERM prediction API.

The request describes a **case**, not a feature vector. A caller filing a PERM
application knows the job, the wages, and the worksite; they do not know
``soc_major_group`` or ``wage_ratio``. Everything derived is computed
server-side by the same code the training pipeline used, which is the only way
to keep training and serving from drifting apart.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data_prep.reference import (
    STATE_NAME_TO_ABBREV,
    VALID_STATE_ABBREVS,
    WAGE_UNIT_TO_ANNUAL,
)

WageUnit = Literal["Year", "Hour", "Month", "Week", "Bi-Weekly"]
EducationLevel = Literal[
    "None", "High School", "Associate's", "Bachelor's", "Master's", "Doctorate", "Other"
]
SkillLevel = Literal["Level I", "Level II", "Level III", "Level IV"]

_SOC_RE = re.compile(r"^\d{2}-?\d{4}(\.\d{2})?$")

# Filing dates outside this window are almost certainly typos rather than real
# cases; PERM as an electronic program does not predate 2005.
MIN_FILING_DATE = date(2005, 1, 1)
MAX_FILING_DATE = date(2035, 12, 31)


class CaseInput(BaseModel):
    """One PERM case, described the way a filer would describe it."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "filing_date": "2024-03-15",
                "soc_code": "15-1252",
                "job_title": "Senior Software Engineer",
                "worksite_state": "CA",
                "prevailing_wage": 145000,
                "prevailing_wage_unit": "Year",
                "offered_wage": 162000,
                "offered_wage_unit": "Year",
                "employer_num_employees": 4200,
                "employer_year_established": 2009,
                "naics_code": "541511",
                "skill_level": "Level III",
                "job_education": "Master's",
                "job_experience_months": 36,
                "class_of_admission": "H-1B",
                "citizenship": "INDIA",
                "has_attorney": True,
            }
        },
    )

    # --- required ----------------------------------------------------------
    filing_date: date = Field(..., description="Date the ETA-9089 was filed.")
    soc_code: str = Field(..., description="SOC/O*NET occupation code, e.g. 15-1252.")
    worksite_state: str = Field(..., description="Two-letter code or full state name.")
    prevailing_wage: float = Field(..., gt=0, description="In prevailing_wage_unit.")
    prevailing_wage_unit: WageUnit = "Year"
    offered_wage: float = Field(..., gt=0, description="In offered_wage_unit.")
    offered_wage_unit: WageUnit = "Year"

    # --- optional employer -------------------------------------------------
    employer_num_employees: int | None = Field(None, ge=1, le=3_000_000)
    employer_year_established: int | None = Field(None, ge=1600, le=2100)
    naics_code: str | None = None
    employer_filing_volume: int = Field(
        0, ge=0,
        description="Filings by this employer seen during training. 0 for an employer "
                    "the model has never encountered, which is how training treats them.",
    )

    # --- optional job ------------------------------------------------------
    job_title: str | None = None
    skill_level: SkillLevel | None = None
    job_education: EducationLevel | None = None
    job_experience_months: int | None = Field(None, ge=0, le=600)
    job_alt_occupation: bool = False
    job_foreign_lang_req: bool = False
    job_combo_occupation: bool = False
    job_req_normal: bool = True
    layoff_past_six_months: bool = False
    professional_occupation: bool = True
    refile: bool = False

    # --- optional worker / representation ----------------------------------
    class_of_admission: str | None = None
    citizenship: str | None = None
    has_attorney: bool = True

    # ----------------------------------------------------------------------
    @field_validator("soc_code")
    @classmethod
    def _check_soc(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not _SOC_RE.match(cleaned):
            raise ValueError(
                f"soc_code {v!r} is not a valid SOC code. Expected NN-NNNN "
                "(e.g. '15-1252'), optionally with an O*NET suffix like '15-1252.00'."
            )
        return cleaned

    @field_validator("worksite_state")
    @classmethod
    def _check_state(cls, v: str) -> str:
        cleaned = v.strip().upper().replace(".", "")
        resolved = STATE_NAME_TO_ABBREV.get(cleaned, cleaned)
        if resolved not in VALID_STATE_ABBREVS:
            raise ValueError(
                f"worksite_state {v!r} is not a recognised US state. Use a two-letter "
                "code (e.g. 'CA') or a full name (e.g. 'California')."
            )
        return resolved

    @field_validator("naics_code")
    @classmethod
    def _check_naics(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = re.sub(r"\D", "", v)
        if not 2 <= len(cleaned) <= 6:
            raise ValueError(f"naics_code {v!r} should be 2-6 digits.")
        return cleaned

    @field_validator("filing_date")
    @classmethod
    def _check_filing_date(cls, v: date) -> date:
        if not MIN_FILING_DATE <= v <= MAX_FILING_DATE:
            raise ValueError(
                f"filing_date {v} is outside the plausible range "
                f"{MIN_FILING_DATE}..{MAX_FILING_DATE}."
            )
        return v

    @model_validator(mode="after")
    def _check_wages(self) -> "CaseInput":
        pw = self.prevailing_wage * WAGE_UNIT_TO_ANNUAL[self.prevailing_wage_unit.upper()]
        ow = self.offered_wage * WAGE_UNIT_TO_ANNUAL[self.offered_wage_unit.upper()]
        for label, annual in (("prevailing_wage", pw), ("offered_wage", ow)):
            if not 10_000 <= annual <= 5_000_000:
                raise ValueError(
                    f"{label} annualises to ${annual:,.0f}, outside the plausible range "
                    f"$10,000-$5,000,000. Check the value against its unit of pay "
                    f"(a yearly figure entered as 'Hour' is the usual cause)."
                )
        return self


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class ModelQuality(BaseModel):
    """How much to trust the number above it.

    Included in every prediction response on purpose. These models are trained on
    synthetic data and are weak in specific, measured ways; returning a bare
    number without that context would misrepresent them.
    """

    trained_on: str = Field(..., description="Dataset the model was fitted to.")
    test_metric: str
    baseline_metric: str
    caveat: str


class ProcessingTimeResponse(BaseModel):
    prediction_id: str
    predicted_days: float
    predicted_months: float
    likely_range_days: list[float] = Field(
        ..., description="Point estimate +/- the model's test MAE. An empirical error "
                         "band, not a calibrated prediction interval."
    )
    filing_date: date
    estimated_decision_date: date
    model_quality: ModelQuality
    input_warnings: list[str] = Field(
        default_factory=list,
        description="Omitted or unrecognised inputs that are materially affecting "
                    "this prediction.")
    explain_url: str


class OutcomeProbability(BaseModel):
    outcome: str
    probability: float


class OutcomeResponse(BaseModel):
    prediction_id: str
    predicted_outcome: str
    confidence: float
    probabilities: list[OutcomeProbability]
    model_quality: ModelQuality
    input_warnings: list[str] = Field(
        default_factory=list,
        description="Omitted or unrecognised inputs that are materially affecting "
                    "this prediction.")
    explain_url: str


class ShapContribution(BaseModel):
    feature: str
    value: str
    shap_value: float
    direction: Literal["increases", "decreases"]


class ExplanationResponse(BaseModel):
    prediction_id: str
    task: Literal["regression", "classification"]
    created_at: str
    prediction: float
    explained_class: str | None = Field(
        None, description="For the classifier, which class these contributions explain."
    )
    base_value: float = Field(
        ..., description="Model output before any feature contribution: the mean "
                         "prediction for regression, the mean class margin for "
                         "classification."
    )
    units: str
    contributions: list[ShapContribution]
    reconstruction_check: float = Field(
        ..., description="base_value + sum(shap_values). Should equal the prediction "
                         "(in margin space for the classifier)."
    )
    note: str


class FieldOption(BaseModel):
    value: str
    label: str


class OptionsResponse(BaseModel):
    """Dropdown choices, derived from the fitted model's category levels."""

    soc_codes: list[FieldOption]
    worksite_states: list[FieldOption]
    skill_levels: list[FieldOption]
    education_levels: list[FieldOption]
    classes_of_admission: list[FieldOption]
    citizenships: list[FieldOption]
    wage_units: list[FieldOption]
    outcome_classes: list[str]


class ServiceInfo(BaseModel):
    """Returned from GET / so a visitor landing on the base URL knows where they are."""

    service: str
    version: str
    environment: str
    docs: str | None
    endpoints: dict[str, str]
    warning: str


class HealthResponse(BaseModel):
    """Liveness. Always 200 while the process serves; `status` carries the nuance."""

    status: str = Field(..., description="'ok' when models are loaded, else 'degraded'.")
    version: str
    environment: str
    models_loaded: list[str]
    stored_predictions: int
    store_capacity: int
    warning: str


class ReadinessResponse(BaseModel):
    """Readiness. 503 when the models are not loaded and predictions cannot be served."""

    ready: bool
    models_loaded: list[str]
    errors: list[str] = Field(default_factory=list)
    artifact_dir: str
