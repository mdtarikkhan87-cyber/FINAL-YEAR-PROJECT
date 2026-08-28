"""End-to-end PERM data preparation pipeline.

    python -m src.data_prep.build_dataset

Load raw disclosure files -> resolve and validate the schema -> clean and
standardise fields -> engineer targets and features -> split chronologically ->
write train/test to ``data/processed/``.

The pipeline is written against the real DOL schema, not the synthetic sample.
It reads whatever CSV/Excel files sit in ``data/raw/``, resolves their columns
through ``schema.COLUMN_ALIASES``, and fails loudly with a named list of missing
columns if a file does not match. Dropping in real disclosure workbooks should
require no code change beyond adding any newly-seen column spelling.

Every row removed is counted and reported. Nothing is dropped silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import cleaning as cl
from . import features as ft
from . import split as sp
from .schema import SchemaError, resolve_columns, validate_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

READABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}


@dataclass
class PrepReport:
    """Row accounting and notes gathered as the pipeline runs."""

    source_files: list[str] = field(default_factory=list)
    rows_loaded: int = 0
    steps: list[tuple[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def drop(self, label: str, n: int) -> None:
        if n:
            self.steps.append((label, n))

    def note(self, text: str) -> None:
        self.notes.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=True, low_memory=False)
    return pd.read_excel(path, dtype=str)


def load_raw(raw_dir: Path, pattern: str | None, report: PrepReport) -> pd.DataFrame:
    """Read every readable file in ``raw_dir`` and concatenate them."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw data directory does not exist: {raw_dir}")

    candidates = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in READABLE_SUFFIXES
        and not p.name.startswith((".", "_"))
        and (pattern is None or p.match(pattern))
    )
    if not candidates:
        raise FileNotFoundError(
            f"no CSV or Excel files found in {raw_dir}.\n"
            "Generate the synthetic sample first:\n"
            "    python -m src.data_prep.make_synthetic\n"
            "or place real DOL disclosure files there (see data/README.md)."
        )

    frames = []
    for path in candidates:
        frame = _read_one(path)
        frame["source_file"] = path.name
        frames.append(frame)
        report.source_files.append(f"{path.name} ({len(frame):,} rows)")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    report.rows_loaded = len(combined)
    return combined


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
def clean_frame(df: pd.DataFrame, report: PrepReport) -> pd.DataFrame:
    """Apply every field-level standardisation rule."""
    out = df.copy()

    # --- dates ---------------------------------------------------------------
    for col in ("filing_date", "decision_date"):
        out[col] = cl.parse_dates(out[col])
    fmt_note = (
        f"dates parsed across mixed formats; "
        f"filing {out['filing_date'].min():%Y-%m-%d} to {out['filing_date'].max():%Y-%m-%d}"
    )
    report.note(fmt_note)

    # --- employer ------------------------------------------------------------
    out["employer_name"] = cl.clean_employer_name(out["employer_name"])
    out["employer_key"] = cl.employer_key(out["employer_name"])
    raw_names = out["employer_name"].nunique(dropna=True)
    keys = out["employer_key"].nunique(dropna=True)
    report.note(
        f"employer names: {raw_names:,} distinct spellings -> {keys:,} distinct keys "
        f"after suffix/punctuation normalisation"
    )

    if "employer_num_employees" in out:
        out["employer_num_employees"] = cl.clean_count(
            out["employer_num_employees"], minimum=1, maximum=3_000_000
        )
    if "employer_year_established" in out:
        out["employer_year_established"] = cl.clean_count(
            out["employer_year_established"], minimum=1600, maximum=2100
        )

    # --- occupation ----------------------------------------------------------
    out["soc_code"] = cl.standardise_soc(out["soc_code"])
    out["soc_code"], remapped = cl.crosswalk_soc_2010_to_2018(out["soc_code"])
    out["soc_major_group"] = cl.soc_major_group(out["soc_code"])
    report.note(
        f"SOC crosswalk 2010->2018 applied to {int(remapped.sum()):,} rows "
        f"({100 * remapped.mean():.1f}%)"
    )

    for col in ("soc_title", "job_title", "skill_level", "job_education",
                "citizenship", "class_of_admission", "worksite_city"):
        if col in out:
            out[col] = cl.clean_text(out[col])

    # --- geography -----------------------------------------------------------
    out["worksite_state"] = cl.standardise_state(out["worksite_state"])
    if "employer_state" in out:
        out["employer_state"] = cl.standardise_state(out["employer_state"])

    # --- wages ---------------------------------------------------------------
    out["prevailing_wage_unit"] = cl.standardise_wage_unit(out["prevailing_wage_unit"])
    out["offered_wage_unit"] = cl.standardise_wage_unit(out["offered_wage_unit"])
    out["prevailing_wage_annual"] = cl.annualise_wage(
        out["prevailing_wage"], out["prevailing_wage_unit"]
    )
    out["offered_wage_annual"] = cl.annualise_wage(
        out["offered_wage"], out["offered_wage_unit"]
    )
    mismatch = (
        out["prevailing_wage_unit"].notna() & out["offered_wage_unit"].notna()
        & (out["prevailing_wage_unit"] != out["offered_wage_unit"])
    )
    report.note(
        f"wages annualised via each row's own unit of pay; "
        f"{int(mismatch.sum()):,} rows ({100 * mismatch.mean():.2f}%) state the offered "
        f"and prevailing wage in different units"
    )
    bad_wage = out["prevailing_wage_annual"].isna() | out["offered_wage_annual"].isna()
    if bad_wage.any():
        report.note(
            f"{int(bad_wage.sum()):,} rows have an unusable wage after annualisation "
            f"(missing unit or implausible amount); kept with NA wage features"
        )

    # --- booleans ------------------------------------------------------------
    for col in ("refile", "job_alt_occupation", "job_foreign_lang_req",
                "job_req_normal", "job_combo_occupation", "layoff_past_six_months",
                "professional_occupation", "job_experience"):
        if col in out:
            out[col] = cl.yes_no_to_bool(out[col])

    if "job_experience_months" in out:
        out["job_experience_months"] = cl.clean_count(
            out["job_experience_months"], minimum=0, maximum=600
        )

    return out


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def write_split(result: sp.SplitResult, out_dir: Path, fmt: str,
                report: PrepReport) -> dict[str, str]:
    """Write train/test, choosing parquet when an engine is installed."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "auto":
        fmt = "parquet" if _parquet_available() else "csv"
        if fmt == "csv":
            report.warn(
                "no parquet engine installed (pyarrow/fastparquet) - wrote CSV instead. "
                "`pip install pyarrow` for smaller, faster, dtype-preserving files."
            )
    elif fmt == "parquet" and not _parquet_available():
        raise RuntimeError(
            "--format parquet requested but neither pyarrow nor fastparquet is "
            "installed. Run `pip install pyarrow` or use --format csv."
        )

    written: dict[str, str] = {}
    for name, frame in (("train", result.train), ("test", result.test)):
        path = out_dir / f"{name}.{fmt}"
        if fmt == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        written[name] = str(path)
    return written


def write_metadata(result: sp.SplitResult, written: dict[str, str], out_dir: Path,
                   report: PrepReport, args: argparse.Namespace) -> Path:
    """Record provenance and the split definition next to the data."""
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": report.source_files,
        "rows_loaded": report.rows_loaded,
        "rows_dropped": {label: n for label, n in report.steps},
        "rows_train": len(result.train),
        "rows_test": len(result.test),
        "split": {
            "split_on": result.split_on,
            "year_column": result.year_column,
            "train_years": result.train_years,
            "test_years": result.test_years,
            "dropped_incomplete_cohorts": result.dropped_incomplete,
            "test_years_requested": args.test_years,
        },
        "cohort_diagnostics": result.diagnostics.to_dict(orient="records"),
        "feature_columns": [c for c in ft.FEATURE_COLUMNS if c in result.train.columns],
        "target_columns": [c for c in ft.TARGET_COLUMNS if c in result.train.columns],
        "passthrough_columns": [c for c in ft.PASSTHROUGH_COLUMNS
                                if c in result.train.columns],
        "notes": report.notes,
        "warnings": report.warnings,
        "files": written,
    }
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(result: sp.SplitResult, written: dict[str, str], meta_path: Path,
                 report: PrepReport) -> None:
    bar = "=" * 78
    print(bar)
    print("PERM DATA PREPARATION")
    print(bar)

    print("\nSOURCE")
    for line in report.source_files:
        print(f"  {line}")
    print(f"  rows loaded: {report.rows_loaded:,}")

    print("\nCLEANING NOTES")
    for line in report.notes:
        print(f"  - {line}")

    if report.steps:
        print("\nROWS DROPPED")
        total = 0
        for label, n in report.steps:
            print(f"  {label:<34} {n:>8,}")
            total += n
        print(f"  {'TOTAL':<34} {total:>8,}  "
              f"({100 * total / max(report.rows_loaded, 1):.2f}% of loaded)")

    print(f"\nCOHORT COMPLETENESS (by {result.split_on} fiscal year)")
    print(sp.format_diagnostics(result.diagnostics, result.split_on))

    print("\nTEMPORAL SPLIT")
    print(f"  split on          : {result.split_on} fiscal year "
          f"({result.year_column})")
    print(f"  train years       : {result.train_years[0]}-{result.train_years[-1]} "
          f"({len(result.train_years)} cohorts)")
    print(f"  test years        : {result.test_years[0]}-{result.test_years[-1]} "
          f"({len(result.test_years)} cohorts)")
    if result.dropped_incomplete:
        print(f"  dropped cohorts   : {result.dropped_incomplete} (incomplete)")
    print(f"  train shape       : {result.train.shape[0]:,} x {result.train.shape[1]}")
    print(f"  test shape        : {result.test.shape[0]:,} x {result.test.shape[1]}")
    print(f"  test fraction     : {100 * len(result.test) / (len(result.train) + len(result.test)):.1f}%")

    year_col = result.year_column
    print("\n  no overlap check:")
    print(f"    max train {year_col} = {int(result.train[year_col].max())}")
    print(f"    min test  {year_col} = {int(result.test[year_col].min())}")
    overlap = set(result.train[year_col]) & set(result.test[year_col])
    print(f"    overlapping years  = {sorted(overlap) if overlap else 'none'}")

    print("\n  actual date ranges:")
    for name, frame in (("train", result.train), ("test", result.test)):
        print(f"    {name:<5} filing   {frame['filing_date'].min():%Y-%m-%d} .. "
              f"{frame['filing_date'].max():%Y-%m-%d}")
        print(f"    {name:<5} decision {frame['decision_date'].min():%Y-%m-%d} .. "
              f"{frame['decision_date'].max():%Y-%m-%d}")

    print("\nTARGETS")
    for name, frame in (("train", result.train), ("test", result.test)):
        d = frame["processing_days"]
        print(f"  {name:<5} processing_days  mean {d.mean():7.1f}  median {d.median():6.0f}"
              f"  p90 {d.quantile(0.9):6.0f}  max {d.max():6.0f}")
    print()
    mix = pd.concat([
        result.train["outcome"].value_counts(normalize=True).rename("train_%") * 100,
        result.test["outcome"].value_counts(normalize=True).rename("test_%") * 100,
    ], axis=1).round(2)
    print(mix.to_string())

    if report.warnings:
        print("\nWARNINGS")
        for line in report.warnings:
            print(f"  ! {line}")

    print("\nWRITTEN")
    for name, path in written.items():
        size = Path(path).stat().st_size / 1_048_576
        print(f"  {name:<6} {path}  ({size:.1f} MB)")
    print(f"  meta   {meta_path}")
    print(bar)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> sp.SplitResult:
    report = PrepReport()

    raw = load_raw(args.raw_dir, args.pattern, report)

    resolved, mapping, unmapped = resolve_columns(raw)
    report.note(f"resolved {len(mapping)} of {len(raw.columns)} raw columns onto the "
                f"canonical schema")
    if unmapped:
        preview = ", ".join(unmapped[:8]) + (" ..." if len(unmapped) > 8 else "")
        report.note(f"{len(unmapped)} unrecognised column(s) dropped: {preview}")

    absent_optional = validate_schema(resolved, source=", ".join(report.source_files))
    if absent_optional:
        report.warn(
            f"{len(absent_optional)} expected-but-optional column(s) absent: "
            f"{', '.join(absent_optional)}"
        )

    cleaned = clean_frame(resolved, report)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["case_number"], keep="first")
    report.drop("duplicate case_number", before - len(cleaned))

    targeted, target_counts = ft.build_targets(cleaned)
    for label in ("dropped_unparseable_dates", "dropped_decision_before_filing",
                  "dropped_implausible_duration", "dropped_non_final_status",
                  "dropped_unknown_status"):
        report.drop(label.replace("dropped_", ""), target_counts.get(label, 0))
    if target_counts.get("_unknown_status_examples"):
        report.warn(
            "unrecognised case_status values dropped: "
            f"{target_counts['_unknown_status_examples']} - add them to "
            "reference.CASE_STATUS_MAP if they are real outcomes"
        )

    featured, feature_counts = ft.build_features(targeted)
    if feature_counts.get("worksite_state_unmapped"):
        report.warn(
            f"{feature_counts['worksite_state_unmapped']:,} rows have an unmappable "
            "worksite_state (kept, region will be NA)"
        )

    final = ft.select_output_columns(featured)

    result = sp.temporal_split(
        final, split_on=args.split_on, test_years=args.test_years,
        drop_incomplete_cohorts=args.drop_incomplete_cohorts,
    )
    result = sp.add_employer_filing_volume(result)

    written = write_split(result, args.out_dir, args.format, report)
    meta_path = write_metadata(result, written, args.out_dir, report, args)

    if args.preview:
        print_report(result, written, meta_path, report)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build train/test datasets from raw PERM disclosure files.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--pattern", default=None,
                        help="glob restricting which files in --raw-dir are read")
    parser.add_argument("--split-on", choices=sorted(sp.SPLIT_KEYS), default="filing",
                        help="fiscal year to split on (default: filing; see split.py "
                             "for why 'decision' avoids censored cohorts)")
    parser.add_argument("--test-years", type=int, default=1,
                        help="how many of the most recent years form the test set")
    parser.add_argument("--drop-incomplete-cohorts", action="store_true",
                        help="exclude cohorts whose observation window is censored")
    parser.add_argument("--format", choices=["auto", "parquet", "csv"], default="auto")
    parser.add_argument("--no-preview", dest="preview", action="store_false")
    args = parser.parse_args(argv)

    try:
        run(args)
    except SchemaError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except (FileNotFoundError, sp.SplitError, RuntimeError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
