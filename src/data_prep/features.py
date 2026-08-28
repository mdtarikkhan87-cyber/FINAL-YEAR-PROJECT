"""Target engineering and derived features.

Two rules govern everything here:

1. **No post-decision information may become a feature.** ``decision_date``
   defines the regression target and ``case_status`` is the classification
   target; neither, nor anything derived from them, appears in the feature list.
   ``FEATURE_COLUMNS`` is the allow-list that enforces this.

2. **Nothing is dropped silently.** Every function returns a counts dict
   alongside the frame so the caller can print a full row-accounting trail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import cleaning as cl
from .reference import STATE_TO_REGION

# Employer headcount bands. Bucketing rather than using the raw count both
# tolerates the field's heavy missingness and gives the fairness audit a
# readable grouping variable.
EMPLOYER_SIZE_BINS = [0, 50, 250, 1_000, 5_000, 25_000, np.inf]
EMPLOYER_SIZE_LABELS = ["1-49", "50-249", "250-999", "1k-4.9k", "5k-24.9k", "25k+"]

# A case taking longer than this is almost certainly a data error rather than a
# slow case; the synthetic generator caps at 1,400 days and real PERM cases do
# not credibly run past ~5 years.
MAX_PLAUSIBLE_PROCESSING_DAYS = 1_825

# The allow-list. Anything not named here never reaches a model.
FEATURE_COLUMNS: tuple[str, ...] = (
    # employer
    "employer_size_bucket", "employer_num_employees", "employer_filing_volume",
    "employer_age_years", "naics_sector",
    # occupation
    "soc_code", "soc_major_group", "skill_level",
    # geography
    "worksite_state", "worksite_region",
    # wages
    "prevailing_wage_annual", "offered_wage_annual", "wage_ratio", "wage_premium",
    "offered_below_prevailing",
    # job requirements
    "job_education", "job_experience_months", "requires_experience",
    "job_alt_occupation", "job_foreign_lang_req", "job_combo_occupation",
    "job_req_normal", "layoff_past_six_months", "professional_occupation", "refile",
    # filing context
    "filing_fiscal_year", "filing_fiscal_quarter", "filing_month",
    # representation
    "has_attorney",
    # foreign worker (see note in HANDOVER.md -- sensitive, include deliberately)
    "class_of_admission", "citizenship",
)

TARGET_COLUMNS: tuple[str, ...] = ("processing_days", "outcome", "is_certified")

# Carried through for grouping, auditing and traceability, but never features.
PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "case_number", "employer_name", "employer_key", "job_title",
    "filing_date", "decision_date", "decision_fiscal_year", "case_status_raw",
)


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Derive ``processing_days``, ``outcome`` and ``is_certified``.

    Drops rows that cannot support a target: unparseable dates, a decision that
    precedes the filing, an implausible duration, or a status that is missing,
    non-final, or unrecognised.
    """
    out = df.copy()
    counts: dict[str, int] = {"rows_in": len(out)}

    # --- regression target ---------------------------------------------------
    out["processing_days"] = (out["decision_date"] - out["filing_date"]).dt.days

    bad_dates = out["filing_date"].isna() | out["decision_date"].isna()
    counts["dropped_unparseable_dates"] = int(bad_dates.sum())
    out = out[~bad_dates]

    non_positive = out["processing_days"] <= 0
    counts["dropped_decision_before_filing"] = int(non_positive.sum())
    out = out[~non_positive]

    too_long = out["processing_days"] > MAX_PLAUSIBLE_PROCESSING_DAYS
    counts["dropped_implausible_duration"] = int(too_long.sum())
    out = out[~too_long]

    # --- classification target -----------------------------------------------
    out["case_status_raw"] = out["case_status"]
    outcome, non_final = cl.standardise_case_status(out["case_status"])
    out["outcome"] = outcome

    counts["dropped_non_final_status"] = int(non_final.sum())
    out = out[~non_final.reindex(out.index, fill_value=False)]

    unknown = out["outcome"].isna()
    counts["dropped_unknown_status"] = int(unknown.sum())
    if unknown.any():
        counts["_unknown_status_examples"] = (
            out.loc[unknown, "case_status_raw"].dropna().unique()[:5].tolist()
        )
    out = out[~unknown]

    # Binary view. Certified-Expired means DOL *did* certify and the employer
    # then let the certification lapse, so it counts as an approval for "will
    # DOL approve this?" and as a failure for "did this case succeed?". The
    # four-class `outcome` keeps both readings available; this column takes the
    # approval reading and the choice is recorded in HANDOVER.md.
    out["is_certified"] = out["outcome"].isin(["certified", "certified_expired"])

    out = out.drop(columns=["case_status"])
    counts["rows_out"] = len(out)
    return out, counts


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Derive the modelling features from already-cleaned canonical columns."""
    out = df.copy()
    counts: dict[str, int] = {}

    # --- employer ------------------------------------------------------------
    if "employer_num_employees" in out:
        out["employer_size_bucket"] = pd.cut(
            out["employer_num_employees"], bins=EMPLOYER_SIZE_BINS,
            labels=EMPLOYER_SIZE_LABELS, right=False,
        ).astype("string")
        counts["employer_size_missing"] = int(out["employer_num_employees"].isna().sum())
    else:
        out["employer_size_bucket"] = pd.NA
        out["employer_num_employees"] = np.nan
        counts["employer_size_missing"] = len(out)

    if "employer_year_established" in out:
        out["employer_age_years"] = (
            out["filing_date"].dt.year - out["employer_year_established"]
        ).where(lambda s: (s >= 0) & (s <= 250))
    else:
        out["employer_age_years"] = np.nan

    out["naics_sector"] = (
        out["naics_code"].astype("string").str.slice(0, 2)
        if "naics_code" in out else pd.NA
    )

    # employer_filing_volume is deliberately NOT computed here -- counting an
    # employer's filings across the whole dataset would leak test-period
    # information into training. It is fitted on the training split only, in
    # split.add_employer_filing_volume.
    out["employer_filing_volume"] = np.nan

    # --- geography -----------------------------------------------------------
    out["worksite_region"] = out["worksite_state"].map(STATE_TO_REGION).astype("string")
    counts["worksite_state_unmapped"] = int(out["worksite_state"].isna().sum())

    # --- wages ---------------------------------------------------------------
    out["wage_ratio"] = out["offered_wage_annual"] / out["prevailing_wage_annual"]
    out["wage_ratio"] = out["wage_ratio"].replace([np.inf, -np.inf], np.nan)
    out["wage_premium"] = out["offered_wage_annual"] - out["prevailing_wage_annual"]
    out["offered_below_prevailing"] = (out["wage_ratio"] < 1.0).astype("boolean")
    out.loc[out["wage_ratio"].isna(), "offered_below_prevailing"] = pd.NA
    counts["wage_ratio_missing"] = int(out["wage_ratio"].isna().sum())

    # --- job requirements ----------------------------------------------------
    if "job_experience_months" in out:
        out["requires_experience"] = (
            out["job_experience_months"].fillna(0) > 0
        ).astype("boolean")
    else:
        out["job_experience_months"] = np.nan
        out["requires_experience"] = pd.NA

    # --- filing context ------------------------------------------------------
    out["filing_fiscal_year"] = cl.fiscal_year(out["filing_date"])
    out["filing_fiscal_quarter"] = cl.fiscal_quarter(out["filing_date"])
    out["filing_month"] = out["filing_date"].dt.month.astype("Int64")
    out["decision_fiscal_year"] = cl.fiscal_year(out["decision_date"])

    # --- representation ------------------------------------------------------
    out["has_attorney"] = (
        out["attorney_name"].notna() if "attorney_name" in out
        else pd.Series(pd.NA, index=out.index, dtype="boolean")
    )

    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
            counts[f"absent_{col}"] = len(out)

    return out, counts


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Order the frame as passthrough, features, targets -- dropping the rest."""
    cols = (
        [c for c in PASSTHROUGH_COLUMNS if c in df.columns]
        + [c for c in FEATURE_COLUMNS if c in df.columns]
        + [c for c in TARGET_COLUMNS if c in df.columns]
    )
    seen: set[str] = set()
    ordered = [c for c in cols if not (c in seen or seen.add(c))]
    return df[ordered]
