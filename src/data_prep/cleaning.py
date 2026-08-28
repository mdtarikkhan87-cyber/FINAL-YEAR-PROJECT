"""Field-level cleaning and standardisation.

Every function here is vectorised over a Series and returns a new Series; none
mutate in place. Rows are never dropped at this level — cleaning turns
unusable values into NA and lets the caller decide what to do about them, so
the row accounting stays in one place.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .reference import (
    CASE_STATUS_MAP,
    MAX_PLAUSIBLE_ANNUAL_WAGE,
    MIN_PLAUSIBLE_ANNUAL_WAGE,
    NON_FINAL_STATUSES,
    SOC_2010_TO_2018,
    STATE_NAME_TO_ABBREV,
    VALID_STATE_ABBREVS,
    WAGE_UNIT_TO_ANNUAL,
)

# Legal-form suffixes stripped when building an employer matching key.
_LEGAL_SUFFIXES = (
    "INCORPORATED", "INCORPORATION", "CORPORATION", "COMPANY", "LIMITED",
    "HOLDINGS", "INC", "LLC", "LLP", "LP", "PLLC", "PC", "PA", "CORP", "CO",
    "LTD", "NA", "SA", "GMBH", "AG", "BV", "NV", "USA",
)
_SUFFIX_RE = re.compile(r"\b(" + "|".join(_LEGAL_SUFFIXES) + r")\b")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9 ]+")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def clean_text(s: pd.Series) -> pd.Series:
    """Trim, collapse internal whitespace, uppercase. Empty strings become NA."""
    out = s.astype("string").str.strip().str.upper()
    out = out.str.replace(_WS_RE, " ", regex=True)
    return out.replace({"": pd.NA, "N/A": pd.NA, "NA": pd.NA, "NONE": pd.NA,
                        "NULL": pd.NA, "UNKNOWN": pd.NA})


def clean_employer_name(s: pd.Series) -> pd.Series:
    """Display form of the employer name: trimmed, uppercased, punctuation-light."""
    out = clean_text(s)
    out = out.str.replace(r"[.,]+", "", regex=True)
    return out.str.replace(_WS_RE, " ", regex=True).str.strip()


def employer_key(s: pd.Series) -> pd.Series:
    """Aggressive matching key for grouping one employer's filings together.

    Drops punctuation and legal-form suffixes so `ACME CORP`, `Acme Corp.` and
    `ACME CORPORATION` collapse to `ACME`. This is a blunt instrument -- it will
    merge genuinely distinct companies that share a base name -- but it is far
    better than treating every spelling as a separate employer. Swap in a real
    entity-resolution step if employer-level features become load-bearing.
    """
    out = clean_text(s).fillna("")
    out = out.str.replace(_NON_ALNUM_RE, " ", regex=True)
    out = out.str.replace(_SUFFIX_RE, " ", regex=True)
    out = out.str.replace(_WS_RE, " ", regex=True).str.strip()
    return out.replace({"": pd.NA})


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
# Formats are tried in order before falling back to pandas' mixed parser.
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y",
                 "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S")


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse a date column that may mix formats between fiscal years.

    Disclosure extracts switch format across years (and the synthetic sample
    reproduces that), so a single `format=` would silently null out half the
    file. Each candidate format is applied to whatever is still unparsed, and
    anything left over goes through the mixed parser.
    """
    raw = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    for fmt in _DATE_FORMATS:
        todo = out.isna() & raw.notna()
        if not todo.any():
            break
        parsed = pd.to_datetime(raw[todo], format=fmt, errors="coerce")
        out.loc[todo] = parsed

    todo = out.isna() & raw.notna()
    if todo.any():
        with pd.option_context("mode.chained_assignment", None):
            out.loc[todo] = pd.to_datetime(raw[todo], format="mixed",
                                           errors="coerce", dayfirst=False)
    return out


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
def standardise_state(s: pd.Series) -> pd.Series:
    """Map states to 2-letter codes, whether given as codes or full names."""
    out = clean_text(s)
    out = out.str.replace(r"[.]", "", regex=True).str.strip()
    mapped = out.map(STATE_NAME_TO_ABBREV)
    out = mapped.fillna(out)
    return out.where(out.isin(VALID_STATE_ABBREVS), pd.NA)


# ---------------------------------------------------------------------------
# SOC codes
# ---------------------------------------------------------------------------
_SOC_DIGITS_RE = re.compile(r"(\d{2})-?(\d{4})")


def standardise_soc(s: pd.Series) -> pd.Series:
    """Normalise SOC codes to the canonical ``NN-NNNN`` form.

    Handles the spellings that turn up in real files: ``15-1132``, ``151132``,
    ``15-1132.00`` (O*NET detail suffix) and stray whitespace.
    """
    text = clean_text(s).fillna("")
    extracted = text.str.extract(_SOC_DIGITS_RE)
    out = extracted[0] + "-" + extracted[1]
    return out.astype("string").replace({"": pd.NA})


def crosswalk_soc_2010_to_2018(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Map 2010 SOC codes onto their 2018 successors.

    Returns the harmonised code and a boolean flag marking which rows were
    remapped. Codes absent from the crosswalk pass through unchanged -- see the
    caveat in ``reference.SOC_2010_TO_2018``.
    """
    mapped = s.map(SOC_2010_TO_2018)
    changed = mapped.notna() & (mapped != s)
    return mapped.fillna(s).astype("string"), changed.fillna(False)


def soc_major_group(s: pd.Series) -> pd.Series:
    """First two digits of the SOC code -- stable across the 2018 revision."""
    return s.astype("string").str.slice(0, 2).replace({"": pd.NA})


# ---------------------------------------------------------------------------
# Wages
# ---------------------------------------------------------------------------
_CURRENCY_RE = re.compile(r"[^0-9.\-]")


def parse_wage_amount(s: pd.Series) -> pd.Series:
    """Coerce a wage column to float, tolerating ``$``, thousands separators."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    text = s.astype("string").str.strip()
    text = text.str.replace(_CURRENCY_RE, "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def standardise_wage_unit(s: pd.Series) -> pd.Series:
    """Fold unit-of-pay spellings onto the canonical uppercase keys."""
    out = clean_text(s)
    out = out.str.replace(r"[_/]", "-", regex=True).str.strip()
    return out.where(out.isin(WAGE_UNIT_TO_ANNUAL.keys()), pd.NA)


def annualise_wage(amount: pd.Series, unit: pd.Series) -> pd.Series:
    """Convert a wage to an annual figure using its own row's unit of pay.

    This is the step that makes wages comparable at all. ``PW_AMOUNT_9089`` is
    meaningless without ``PW_UNIT_OF_PAY_9089``: the same column holds 120000
    (Year) and 57.69 (Hour). Rows whose unit is missing or unrecognised return
    NA rather than being silently treated as annual.
    """
    factor = unit.map(WAGE_UNIT_TO_ANNUAL)
    annual = parse_wage_amount(amount) * pd.to_numeric(factor, errors="coerce")
    implausible = (annual < MIN_PLAUSIBLE_ANNUAL_WAGE) | (annual > MAX_PLAUSIBLE_ANNUAL_WAGE)
    return annual.mask(implausible)


# ---------------------------------------------------------------------------
# Case status
# ---------------------------------------------------------------------------
def standardise_case_status(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Fold case-status spellings onto four canonical outcomes.

    Returns the canonical outcome and a boolean marking rows whose status was
    non-final (pending, in process). Ordering matters: ``CERTIFIED-EXPIRED``
    must not be matched by a prefix test for ``CERTIFIED``, so this is an exact
    lookup on a normalised key, never a substring test.
    """
    key = clean_text(s)
    key = key.str.replace(r"[\s_/]+", "-", regex=True).str.strip("-")
    non_final = key.isin({k.replace(" ", "-") for k in NON_FINAL_STATUSES} | NON_FINAL_STATUSES)
    outcome = key.map(CASE_STATUS_MAP).astype("string")
    return outcome, non_final.fillna(False)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def yes_no_to_bool(s: pd.Series) -> pd.Series:
    """Map Y/N/YES/NO/TRUE/FALSE/1/0 to a nullable boolean."""
    text = clean_text(s)
    mapping = {"Y": True, "YES": True, "TRUE": True, "T": True, "1": True,
               "N": False, "NO": False, "FALSE": False, "F": False, "0": False}
    return text.map(mapping).astype("boolean")


def clean_count(s: pd.Series, minimum: float = 0.0,
                maximum: float | None = None) -> pd.Series:
    """Numeric coercion with an out-of-range guard, returning NA when implausible."""
    out = pd.to_numeric(
        s.astype("string").str.replace(_CURRENCY_RE, "", regex=True)
        if not pd.api.types.is_numeric_dtype(s) else s,
        errors="coerce",
    )
    out = out.mask(out < minimum)
    if maximum is not None:
        out = out.mask(out > maximum)
    return out


def fiscal_year(dates: pd.Series) -> pd.Series:
    """U.S. federal fiscal year: 1 Oct (Y-1) through 30 Sep (Y) is FY Y."""
    d = pd.to_datetime(dates, errors="coerce")
    return (d.dt.year + (d.dt.month >= 10).astype("Int64")).astype("Int64")


def fiscal_quarter(dates: pd.Series) -> pd.Series:
    """Federal fiscal quarter, 1-4, where Oct-Dec is Q1."""
    d = pd.to_datetime(dates, errors="coerce")
    return (((d.dt.month - 10) % 12) // 3 + 1).astype("Int64")
