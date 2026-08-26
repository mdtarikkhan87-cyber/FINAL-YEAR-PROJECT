# Handover — Predicting PERM Visa Processing Time & Outcome

**Last session:** 2026-08-27
**Phase:** 1 of ~11 — scaffold + synthetic data generator complete. No models yet.

---

## 1. Where things are

| | |
|---|---|
| Project root | `C:\Users\mdtar\OneDrive\Desktop\final year project\` |
| Python | 3.12.10 (system install — **no virtualenv created yet**) |
| Installed | pandas 2.3.3, numpy 2.2.6 only. `requirements.txt` has **not** been `pip install`-ed. |
| Version control | **Not a git repo.** `.gitignore` exists but nothing is tracked. `git init` is an open task. |

The project used to live at `Desktop\toks\perm-visa-ml\`. It was moved out on 2026-08-27
because `toks` holds an unrelated frontend/design project. **Do not write to `toks`.**

---

## 2. What exists now

```
final year project/
├── HANDOVER.md                        <- you are here
├── README.md                          problem statement, architecture, stack, setup (setup = placeholder)
├── requirements.txt                   full dep list, not yet installed
├── .gitignore                         ignores data/, reports/, mlruns/, __pycache__, .venv
├── data/
│   ├── README.md                      DOL source, which FYs/quarters to pull, placement rules
│   ├── raw/
│   │   └── perm_synthetic_sample.csv  50,000 x 58, 21 MB  (GITIGNORED — regenerate, don't hunt for it)
│   └── processed/                     empty
├── src/
│   ├── data_prep/
│   │   └── make_synthetic.py          1,213 lines. The only real code in the project.
│   ├── models/                        empty (package stub)
│   ├── explainability/                empty (package stub)
│   ├── fairness/                      empty (package stub)
│   └── validation/                    empty (package stub)
├── api/                               empty (package stub)
├── app/                               empty (package stub)
├── reports/                           empty
└── paper/                             empty
```

Every `src/` subdir, `api/`, and `app/` has an `__init__.py`, so `python -m src.x.y` works.

---

## 3. What was done on 2026-08-27

1. **Scaffolded** the folder structure, `README.md`, `requirements.txt`, `data/README.md`, `.gitignore`.
2. **Moved** the project out of `toks` to its own Desktop folder (files moved, not regenerated).
3. **Wrote and ran** `src/data_prep/make_synthetic.py`, producing `data/raw/perm_synthetic_sample.csv`.

Nothing else was built. No models, no pipeline, no API, no app, no tests.

---

## 4. How to run it

Regenerate the dataset (this is the only runnable thing in the project today):

```bash
python -m src.data_prep.make_synthetic
```

Useful flags:

```bash
python -m src.data_prep.make_synthetic --rows 5000 --no-preview
```

| Flag | Effect |
|---|---|
| `--rows N` | total rows (default 50000) |
| `--employers N` | employer pool size (default 1400) |
| `--years 2018 2019 ...` | decision fiscal years (default 2018–2024) |
| `--seed N` | default 42 — **this is the provenance record for the file** |
| `--no-messy` | emit clean, already-harmonised values |
| `--include-ground-truth` | adds `_SYNTH_AUDITED`, `_SYNTH_PROCESSING_DAYS`, `_SYNTH_SECTOR`. **Leaks the answer** — off by default, use only to validate the pipeline recovers known effects |
| `--no-preview` | skip the printed summary |

Output is a deterministic function of `--seed`, `--rows`, `--employers`, `--years`.

---

## 5. Design decisions already locked in (and why)

These are load-bearing. Changing one changes the validity of everything downstream.

- **Rows are grouped by DECISION fiscal year, not filing year.** Mirrors the real DOL files:
  an FY2023 file contains cases *decided* in FY2023, most of them filed earlier. In the
  synthetic file, 75% of rows were filed in an earlier FY than their `FISCAL_YEAR`.
  **The temporal split must key on decision date.**

- **Temporal validation, not random k-fold.** Backlogs move hard across years (median
  processing time runs 164 days in FY2018 to 417 in FY2023). A random split leaks the future.
  Planned split: train FY2018–21, validate FY2022, test FY2023–24.

- **Pre-decision features only.** `DECISION_DATE` defines the regression target and must never
  be a feature. `CASE_STATUS` is the classification target. Processing days is deliberately
  *not* a column — derive it in the prep layer from the two date columns.

- **Two targets, one feature pipeline.** Processing time (regression) and outcome
  (classification) share nearly all inputs.

- **Audits are latent.** Real disclosure files have no audit flag — you infer it from the
  bimodal processing-time distribution. The generator models audit as a hidden driver of both
  processing time and outcome (~21% rate; median 561 days audited vs 240 not) and does not
  emit it unless `--include-ground-truth`.

---

## 6. Intentional "weirdness" in the synthetic data — do NOT "fix" it

A future session could easily mistake these for defects and clean them up. They are all
deliberate, and they exist so the harmonisation layer gets built properly:

| What you'll see | Why it's there |
|---|---|
| Mixed date formats: `MM/DD/YYYY` for FY≤2020, `YYYY-MM-DD` for FY≥2021 | Real files change format between extracts |
| SOC codes change at FY2020 — `15-1132` becomes `15-1252`, several 2010 codes collapse into one 2018 code | Genuine 2010→2018 SOC revision. Untreated, the same job is two unrelated categories split exactly on the train/test boundary |
| 1,400 employers appear as ~1,877 distinct name strings (casing, `, INC.` vs ` INC`, whitespace) | Entity resolution is a real problem in this data |
| Wage amounts only make sense with their own row's `*_UNIT_OF_PAY_9089` (Year/Hour/Month/Week/Bi-Weekly), and ~1.9% of rows use a *different* unit for offered vs prevailing wage | A raw mean over the wage column is meaningless — annualise first |
| `CASE_STATUS` casing varies; some states spelled out in full; stray whitespace | Extract-to-extract inconsistency |
| Conditional missingness (`ORIG_FILE_DATE` only when `REFILE=Y`, etc.) | Real files blank fields conditionally, not uniformly |

`--no-messy` turns the cosmetic drift off. SOC drift is always on.

**Tuned constants — change with care.** `FY_BACKLOG_MEDIAN_DAYS`, the outcome base logits
(`logit_certified = 3.45`), and the offered-wage premium (`1.0 + lognormal(-2.60, 0.75)`) were
tuned to hit realistic target distributions. Two bugs were found and fixed during validation:
the premium was originally a lognormal centred just above 1.0, which put 27% of offers *below*
prevailing wage instead of 1.6% and drowned the denial signal; and the unit-mismatch redraw
usually redrew the same unit. Current state: below-PW rows are 1.51% and carry a 29.2% denial
rate vs 5.2% baseline.

**Realised distributions:** Certified 88.1% / Denied 5.6% / Withdrawn 4.3% / Certified-Expired 2.0%.
Citizenship: India 59.8%, China 10.9%. Occupations sector-conditioned (hospitals file nurses,
not developers).

---

## 7. Open decisions — these need your call before the next phase

1. **How is "employer size" defined?** The real `EMPLOYER_NUM_EMPLOYEES` field is
   inconsistently populated across fiscal years (8.2% null even in the synthetic file, and worse
   in reality). The alternative is a filing-volume proxy computed per employer. This choice
   shapes both the feature set *and* the fairness audit's group definitions — it can't be
   deferred past feature engineering.

2. **Does `COUNTRY_OF_CITIZENSHIP` go in the features, the fairness audit, or neither?**
   It and `FW_INFO_BIRTH_COUNTRY` are genuine PERM disclosure fields and are in the dataset.
   `README.md` currently frames the fairness audit around structural proxies (employer size,
   occupation, region) on the grounds that the data has no applicant race or gender — true, but
   citizenship is considerably more protected-class-adjacent than those proxies. Whichever way
   this goes, it needs stating explicitly in `paper/`.

3. **Folder name contains a space.** Fine for Python (`python -m src...` works — verified), but
   it needs quoting in every `cd` and may trip up other tooling later. Renaming to
   `final-year-project` is a cheap fix now and an annoying one later.

---

## 8. Next up

Roadmap position — everything below is unstarted:

- [ ] `git init` + first commit
- [ ] Create `.venv`, `pip install -r requirements.txt`
- [ ] **`src/data_prep/build_dataset.py`** — ingest `data/raw/`, harmonise schema across FYs
      (SOC drift, date formats, employer names, wage annualisation), write parquet to
      `data/processed/`. Build it against the synthetic file; it must work on the real files unchanged.
- [ ] `src/validation/` — temporal split framework (split on decision date)
- [ ] Feature engineering — employer size buckets, SOC major group, Census region, wage ratio
      (offered/prevailing, annualised), filing quarter
- [ ] Baseline models, then XGBoost, for both targets; MLflow tracking
- [ ] SHAP, fairlearn audit, FastAPI, Streamlit, Docker, write-up

Suggested next task: `build_dataset.py`. It's the piece everything else waits on, and the
synthetic file was built specifically to make it testable.

---

## 9. Gotchas

- **`data/raw/` and `data/processed/` are gitignored.** The CSV will not survive a clone or a
  clean checkout. Regenerate with `python -m src.data_prep.make_synthetic` (seed 42 reproduces
  the exact file). This is intentional, not an oversight.
- **No real DOL data has been downloaded.** See `data/README.md` for the source, which fiscal
  years/quarters to pull, and the warning that quarterly files are usually *cumulative within a
  fiscal year* — concatenating every quarter duplicates cases.
- **Nothing from the synthetic data is a result.** Its correlations are invented by the
  generator. No accuracy score, SHAP plot, or fairness metric from this file belongs in
  `reports/` or `paper/` without a synthetic-data label.
- The generator prints ASCII only — the Windows console default codepage mangles em dashes.
