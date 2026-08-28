"""Schema resolution and validation for raw PERM disclosure files.

Real DOL disclosure files rename columns between fiscal years, so the pipeline
never refers to a raw column name directly. Instead every raw file is resolved
onto a canonical snake_case schema here, and everything downstream works against
that. Adding a newly-seen spelling means adding one alias below, not editing the
cleaning code.
"""

from __future__ import annotations

import difflib
import re

import pandas as pd


class SchemaError(ValueError):
    """Raised when a raw file cannot be resolved onto the canonical schema."""


# canonical_name -> raw spellings seen (or plausibly used) across fiscal years.
# The canonical name itself is always an accepted alias.
COLUMN_ALIASES: dict[str, list[str]] = {
    # --- identity and dates -------------------------------------------------
    "case_number": ["CASE_NUMBER", "CASE_NO", "CASE_ID"],
    "case_status": ["CASE_STATUS", "STATUS", "CASE_STATUS_9089"],
    "filing_date": ["RECEIVED_DATE", "CASE_RECEIVED_DATE", "FILING_DATE",
                    "RECEIVED_DATE_9089"],
    "decision_date": ["DECISION_DATE", "CASE_DECISION_DATE", "DECISION_DATE_9089"],
    # --- employer -----------------------------------------------------------
    "employer_name": ["EMPLOYER_NAME", "EMPLOYER_NAME_9089", "EMP_NAME"],
    "employer_city": ["EMPLOYER_CITY", "EMPLOYER_CITY_9089"],
    "employer_state": ["EMPLOYER_STATE", "EMPLOYER_STATE_PROVINCE",
                       "EMPLOYER_STATE_9089"],
    "employer_num_employees": ["EMPLOYER_NUM_EMPLOYEES", "EMPLOYER_NUMBER_OF_EMPLOYEES",
                               "EMP_NUM_EMPLOYEES"],
    "employer_year_established": ["EMPLOYER_YEAR_COMMENCED_BUSINESS",
                                  "EMPLOYER_YR_ESTAB"],
    "naics_code": ["NAICS_CODE", "NAIC_CODE", "EMPLOYER_NAICS_CODE"],
    # --- occupation ---------------------------------------------------------
    "soc_code": ["PW_SOC_CODE", "SOC_CODE", "OCCUPATION_CODE", "PW_SOC_CODE_9089",
                 "JOB_INFO_SOC_CODE"],
    "soc_title": ["PW_SOC_TITLE", "SOC_TITLE", "OCCUPATION_TITLE",
                  "PW_SOC_TITLE_9089", "JOB_INFO_SOC_TITLE"],
    "skill_level": ["PW_SKILL_LEVEL", "PW_LEVEL_9089", "PW_SKILL_LEVEL_9089"],
    "job_title": ["JOB_INFO_JOB_TITLE", "JOB_TITLE", "JOB_INFO_JOB_TITLE_9089"],
    # --- wages --------------------------------------------------------------
    "prevailing_wage": ["PW_AMOUNT_9089", "PW_AMOUNT", "PREVAILING_WAGE", "PW_WAGE"],
    "prevailing_wage_unit": ["PW_UNIT_OF_PAY_9089", "PW_UNIT_OF_PAY", "PW_UNIT"],
    "offered_wage": ["WAGE_OFFER_FROM_9089", "WAGE_OFFERED_FROM_9089",
                     "WAGE_OFFER_FROM", "WAGE_OFFERED_FROM"],
    "offered_wage_to": ["WAGE_OFFER_TO_9089", "WAGE_OFFERED_TO_9089",
                        "WAGE_OFFER_TO"],
    "offered_wage_unit": ["WAGE_OFFER_UNIT_OF_PAY_9089",
                          "WAGE_OFFERED_UNIT_OF_PAY_9089", "WAGE_OFFER_UNIT_OF_PAY"],
    "pw_source": ["PW_SOURCE", "PW_SOURCE_NAME_9089", "PW_SOURCE_NAME"],
    # --- worksite -----------------------------------------------------------
    "worksite_city": ["JOB_INFO_WORK_CITY", "WORKSITE_CITY", "JOB_INFO_WORK_CITY_9089"],
    "worksite_state": ["JOB_INFO_WORK_STATE", "WORKSITE_STATE",
                       "JOB_INFO_WORK_STATE_9089"],
    # --- job requirements ---------------------------------------------------
    "job_education": ["JOB_INFO_EDUCATION", "MINIMUM_EDUCATION"],
    "job_experience": ["JOB_INFO_EXPERIENCE"],
    "job_experience_months": ["JOB_INFO_EXPERIENCE_NUM_MONTHS"],
    "job_alt_occupation": ["JOB_INFO_ALT_OCC"],
    "job_foreign_lang_req": ["JOB_INFO_FOREIGN_LANG_REQ"],
    "job_req_normal": ["JOB_INFO_JOB_REQ_NORMAL"],
    "job_combo_occupation": ["JOB_INFO_COMBO_OCCUPATION"],
    "layoff_past_six_months": ["RI_LAYOFF_IN_PAST_SIX_MONTHS",
                               "RECR_INFO_LAYOFF_IN_PAST_SIX_MONTHS"],
    "professional_occupation": ["RECR_INFO_PROFESSIONAL_OCC"],
    "refile": ["REFILE"],
    # --- foreign worker -----------------------------------------------------
    "citizenship": ["COUNTRY_OF_CITIZENSHIP", "COUNTRY_OF_CITZENSHIP",
                    "FW_INFO_CITIZENSHIP"],
    "birth_country": ["FW_INFO_BIRTH_COUNTRY", "FOREIGN_WORKER_BIRTH_COUNTRY"],
    "class_of_admission": ["CLASS_OF_ADMISSION"],
    "worker_education": ["FOREIGN_WORKER_INFO_EDUCATION", "FW_INFO_EDUCATION"],
    # --- representation -----------------------------------------------------
    "attorney_name": ["AGENT_ATTORNEY_NAME", "AGENT_ATTORNEY_FIRM_NAME"],
    "attorney_state": ["AGENT_ATTORNEY_STATE"],
    # --- provenance ---------------------------------------------------------
    "source_fiscal_year": ["FISCAL_YEAR", "FY"],
}

# Without these the pipeline cannot produce its targets or its core features,
# so a file missing any of them is rejected outright.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "case_number",
    "case_status",
    "filing_date",
    "decision_date",
    "employer_name",
    "soc_code",
    "job_title",
    "worksite_state",
    "prevailing_wage",
    "prevailing_wage_unit",
    "offered_wage",
    "offered_wage_unit",
)

# Used when present, imputed or skipped when absent. Their absence is reported
# but never fatal, because coverage genuinely varies by fiscal year.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "employer_num_employees", "employer_state", "employer_city", "naics_code",
    "soc_title", "skill_level", "worksite_city", "job_education",
    "job_experience", "job_experience_months", "refile", "citizenship",
    "class_of_admission", "attorney_name", "layoff_past_six_months",
    "source_fiscal_year",
)


def _normalise_header(name: str) -> str:
    """Fold a raw header to a comparable form: upper, underscores, no padding."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip()).strip("_")
    return cleaned.upper()


def _build_lookup() -> dict[str, str]:
    """Normalised raw spelling -> canonical name."""
    lookup: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in [canonical, *aliases]:
            lookup[_normalise_header(alias)] = canonical
    return lookup


ALIAS_LOOKUP = _build_lookup()


def resolve_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Rename raw columns onto the canonical schema.

    Returns the renamed frame, the mapping that was applied, and the raw columns
    that no other name claimed. Unrecognised columns are dropped rather than
    carried along, so the processed dataset has a stable, documented shape.
    """
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    taken: set[str] = set()

    for raw in df.columns:
        canonical = ALIAS_LOOKUP.get(_normalise_header(raw))
        if canonical is None:
            unmapped.append(str(raw))
        elif canonical in taken:
            # Two raw columns resolved to the same canonical name; keep the
            # first and report the duplicate rather than silently overwriting.
            unmapped.append(f"{raw} (duplicate of {canonical})")
        else:
            mapping[str(raw)] = canonical
            taken.add(canonical)

    resolved = df.rename(columns=mapping)[list(mapping.values())]
    return resolved, mapping, unmapped


def validate_schema(df: pd.DataFrame, source: str = "input") -> list[str]:
    """Check the canonical frame has what the pipeline needs.

    Raises SchemaError naming every missing required column, with a suggestion
    drawn from the columns actually present. Returns the list of expected-but-
    absent optional columns so the caller can report them.
    """
    present = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]

    if missing:
        lines = [
            f"Schema validation failed for {source}.",
            f"{len(missing)} required column(s) could not be resolved:",
            "",
        ]
        for col in missing:
            known = COLUMN_ALIASES.get(col, [])
            suggestion = difflib.get_close_matches(col, sorted(present), n=1, cutoff=0.6)
            hint = f"  closest column present: {suggestion[0]!r}" if suggestion else ""
            lines.append(f"  - {col}")
            lines.append(f"      accepted raw names: {', '.join(known) or 'n/a'}")
            if hint:
                lines.append(hint)
        lines += [
            "",
            f"Columns resolved from the file ({len(present)}): "
            f"{', '.join(sorted(present)) or 'none'}",
            "",
            "Fix by adding the file's spelling to COLUMN_ALIASES in "
            "src/data_prep/schema.py, then re-run.",
        ]
        raise SchemaError("\n".join(lines))

    return [c for c in EXPECTED_COLUMNS if c not in present]
