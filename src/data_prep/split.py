"""Temporal train/test splitting, with cohort-completeness diagnostics.

Random k-fold is invalid on this data. PERM backlogs shift year to year, so a
random split lets the model see the future and reports a score it will never
reproduce in deployment. Every split here is chronological.

Which date to split on is a genuine methodological choice, and both options are
flawed in opposite directions:

**Filing year** matches how a user experiences the problem -- you file today and
want a prediction -- but a fixed data extract only contains *decided* cases, so
the most recent filing cohorts are **right-censored**: their slow cases have not
been decided yet and are simply absent. A test set built from the newest filing
year therefore over-represents fast cases and flatters any processing-time
model. The oldest filing cohorts have the mirror problem, **left-truncation**:
only cases slow enough to still be pending when the data window opened appear.

**Decision year** matches how the data is published (an FY2023 file holds cases
decided in FY2023) and how you would actually deploy -- "I have every file
through FY2023, predict FY2024". No cohort is censored. The cost is that some
training cases were *filed* after some test cases.

``diagnose_cohorts`` quantifies the censoring so the choice can be made on
evidence rather than assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SPLIT_KEYS = {
    "filing": ("filing_fiscal_year", "filing_date"),
    "decision": ("decision_fiscal_year", "decision_date"),
}


class SplitError(ValueError):
    """Raised when a temporal split cannot be formed."""


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    split_on: str
    year_column: str
    train_years: list[int]
    test_years: list[int]
    diagnostics: pd.DataFrame
    dropped_incomplete: list[int] = field(default_factory=list)


def _fy_bounds(fy: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Federal fiscal year FY spans 1 Oct (FY-1) to 30 Sep (FY)."""
    return pd.Timestamp(year=fy - 1, month=10, day=1), pd.Timestamp(year=fy, month=9, day=30)


def diagnose_cohorts(df: pd.DataFrame, split_on: str) -> pd.DataFrame:
    """Per-cohort completeness report.

    For a filing-year cohort, computes the window of processing times the data
    could physically have observed:

    * ``min_observable_days`` -- a case decided faster than this would have been
      decided before the data window opened, so it is missing (left-truncation).
    * ``max_observable_days`` -- a case slower than this has not been decided
      yet, so it is missing (right-censoring).

    A cohort is ``complete`` when its observable window covers the processing
    times actually seen elsewhere in the data (the 99th percentile is used as
    the reference).
    """
    year_col, _ = SPLIT_KEYS[split_on]
    data_min_decision = df["decision_date"].min()
    data_max_decision = df["decision_date"].max()
    reference_p99 = float(df["processing_days"].quantile(0.99))

    rows = []
    for year, grp in df.groupby(year_col, dropna=True):
        year = int(year)
        if split_on == "filing":
            cohort_start, cohort_end = _fy_bounds(year)
            # Optimistic bound: the earliest-filed case has the longest window.
            max_observable = (data_max_decision - cohort_start).days
            # Pessimistic bound: the latest-filed case needed at least this many
            # days to survive into the data window at all.
            min_observable = max(0, (data_min_decision - cohort_end).days)
        else:
            # Decision-year cohorts are complete by construction: the file is
            # grouped by decision date, so every decision in the year is present.
            # A sentinel well above any real duration keeps `complete` True.
            max_observable = 10**6
            min_observable = 0

        complete = (min_observable <= 0) and (max_observable >= reference_p99)
        rows.append({
            "year": year,
            "n": len(grp),
            "pct_of_total": 100.0 * len(grp) / len(df),
            "median_days": float(grp["processing_days"].median()),
            "p90_days": float(grp["processing_days"].quantile(0.90)),
            "max_days": int(grp["processing_days"].max()),
            "min_observable_days": min_observable,
            "max_observable_days": max_observable,
            "complete": complete,
        })

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    out.attrs["reference_p99"] = reference_p99
    return out


def temporal_split(
    df: pd.DataFrame,
    split_on: str = "filing",
    test_years: int = 1,
    drop_incomplete_cohorts: bool = False,
) -> SplitResult:
    """Split chronologically: the newest ``test_years`` cohorts become the test set."""
    if split_on not in SPLIT_KEYS:
        raise SplitError(
            f"split_on must be one of {sorted(SPLIT_KEYS)}, got {split_on!r}"
        )
    year_col, _ = SPLIT_KEYS[split_on]
    if year_col not in df.columns:
        raise SplitError(f"expected column {year_col!r} is missing; run build_features first")

    working = df[df[year_col].notna()].copy()
    working[year_col] = working[year_col].astype(int)

    diagnostics = diagnose_cohorts(working, split_on)

    dropped: list[int] = []
    if drop_incomplete_cohorts:
        incomplete = diagnostics.loc[~diagnostics["complete"], "year"].tolist()
        if incomplete:
            dropped = [int(y) for y in incomplete]
            working = working[~working[year_col].isin(dropped)]

    years = sorted(working[year_col].unique())
    if len(years) < 2:
        raise SplitError(
            f"need at least 2 distinct {split_on} years to split, found {len(years)}: {years}. "
            "Widen the input or lower --test-years."
        )
    if test_years >= len(years):
        raise SplitError(
            f"--test-years={test_years} leaves no training data: only {len(years)} "
            f"{split_on} years available ({years[0]}-{years[-1]})."
        )

    test_year_list = years[-test_years:]
    train_year_list = years[:-test_years]

    train = working[working[year_col].isin(train_year_list)].copy()
    test = working[working[year_col].isin(test_year_list)].copy()

    if train.empty or test.empty:
        raise SplitError(
            f"temporal split produced an empty side (train={len(train)}, test={len(test)})."
        )

    return SplitResult(
        train=train, test=test, split_on=split_on, year_column=year_col,
        train_years=[int(y) for y in train_year_list],
        test_years=[int(y) for y in test_year_list],
        diagnostics=diagnostics, dropped_incomplete=dropped,
    )


def add_employer_filing_volume(result: SplitResult) -> SplitResult:
    """Attach each employer's filing count, counted on the training split only.

    Counting across the whole dataset would leak test-period activity into a
    training feature. Employers not seen during training get 0, which is the
    honest value for a model that has never encountered them.
    """
    if "employer_key" not in result.train.columns:
        return result

    volume = result.train["employer_key"].value_counts()
    for frame in (result.train, result.test):
        frame["employer_filing_volume"] = (
            frame["employer_key"].map(volume).fillna(0).astype("int64")
        )
    return result


def format_diagnostics(diag: pd.DataFrame, split_on: str) -> str:
    """Render the cohort table with a censoring warning when one is warranted."""
    show = diag.copy()
    show["pct_of_total"] = show["pct_of_total"].round(1)
    show["median_days"] = show["median_days"].round(0).astype(int)
    show["p90_days"] = show["p90_days"].round(0).astype(int)
    if split_on == "decision":
        show = show.drop(columns=["min_observable_days", "max_observable_days"])
    lines = [show.to_string(index=False)]

    incomplete = diag.loc[~diag["complete"], "year"].tolist()
    if incomplete:
        p99 = diag.attrs.get("reference_p99", float("nan"))
        lines += [
            "",
            "WARNING: incomplete cohorts detected -> " +
            ", ".join(str(int(y)) for y in incomplete),
            f"  Processing times reach {p99:.0f} days at the 99th percentile, but these",
            "  cohorts had a shorter window in which a case could be observed at all.",
            "  Their slow cases are absent from the extract, not rare in reality, so any",
            "  processing-time metric computed on them is biased downward.",
            "  Re-run with --drop-incomplete-cohorts to exclude them, or with",
            "  --split-on decision, which has no censored cohorts by construction.",
        ]
    return "\n".join(lines)
