# Data

This directory holds every dataset the project touches. **Nothing in `raw/` or
`processed/` is committed to version control** — the DOL files are large and
freely re-downloadable, and processed tables are reproducible from them.

```
data/
├── raw/          # Exactly as downloaded from DOL. Never edit these files.
├── processed/    # Cleaned / feature-engineered outputs (parquet)
└── README.md     # This file
```

---

## 1. Where the real data comes from

**U.S. Department of Labor — Office of Foreign Labor Certification (OFLC)
Performance Data:**

<https://www.dol.gov/agencies/eta/foreign-labor/performance>

That page publishes **quarterly disclosure files** for every OFLC program. We
want the **PERM** program only (not H-1B/LCA, not H-2A, not H-2B, not
PW — Prevailing Wage). Files are Excel workbooks (`.xlsx`, older years `.xls`),
occasionally offered as CSV, typically named along the lines of:

```
PERM_Disclosure_Data_FY2023_Q4.xlsx
PERM_Disclosure_Data_FY2024_Q2.xlsx
```

Each row is one filed PERM application (ETA Form 9089) that reached a final
determination in that fiscal year, with employer, job, wage, and case-status
fields.

> **Fiscal year, not calendar year.** A DOL fiscal year runs **October 1 →
> September 30**. FY2024 Q1 is Oct–Dec 2023.

> **Quarterly files are usually cumulative within a fiscal year.** The Q2 file
> for a given FY generally contains Q1 + Q2, and the Q4 file contains the whole
> year. Verify this on the page before downloading everything — if it holds,
> **one Q4 file per completed fiscal year is enough**, plus the latest available
> quarter for the fiscal year currently in progress. Downloading every quarter
> and concatenating them will duplicate cases.

---

## 2. Which years and quarters to grab

The temporal-validation design needs several *completed* fiscal years, ordered,
with the newest held out.

### Recommended pull

| Fiscal year | File to download | Intended role |
|---|---|---|
| FY2018 | `..._FY2018_Q4` | Train |
| FY2019 | `..._FY2019_Q4` | Train |
| FY2020 | `..._FY2020_Q4` | Train (COVID-affected — see caveat) |
| FY2021 | `..._FY2021_Q4` | Train (COVID-affected — see caveat) |
| FY2022 | `..._FY2022_Q4` | Validation / tuning |
| FY2023 | `..._FY2023_Q4` | Test (held out) |
| FY2024 | `..._FY2024_Q4` | Test (held out) |
| Current FY | latest quarter available | Optional smoke-test slice |

That is roughly **7 fiscal years, ~100k–200k rows each**, on the order of
1M+ total rows and a few hundred MB of Excel. Expect slow first parses; the
prep pipeline converts to parquet once and reads that afterwards.

### Minimum viable pull

If bandwidth or time is short, **FY2021–FY2024 (four files)** is enough to train
and run a genuine temporal split (train FY2021–22, validate FY2023, test
FY2024). More years mainly help the model learn backlog regimes.

### Caveats when choosing years

- **COVID distortion (FY2020–FY2021).** Processing times and audit behavior are
  atypical. Keep the years, but flag them with a feature and check whether
  excluding them changes conclusions.
- **Schema drift.** Column names and codings change between fiscal years (e.g.
  SOC code/title field names, wage-unit encodings, case-status label spellings).
  Harmonization belongs in `src/data_prep/`, not in hand-edited spreadsheets.
- **Older years (pre-FY2018)** exist and can extend the training window, but the
  schema diverges further and the pre-2016 filing process differs enough that
  the added rows may hurt more than help. Add them only after harmonization is
  solid.

---

## 3. Where to put the files

Download and drop the workbooks **unmodified** into `data/raw/`:

```
data/raw/
├── PERM_Disclosure_Data_FY2018_Q4.xlsx
├── PERM_Disclosure_Data_FY2019_Q4.xlsx
├── PERM_Disclosure_Data_FY2020_Q4.xlsx
├── PERM_Disclosure_Data_FY2021_Q4.xlsx
├── PERM_Disclosure_Data_FY2022_Q4.xlsx
├── PERM_Disclosure_Data_FY2023_Q4.xlsx
└── PERM_Disclosure_Data_FY2024_Q4.xlsx
```

Rules for `raw/`:

- **Never edit a file in place.** No opening in Excel and re-saving, no manual
  column fixes, no deleting rows. Every correction happens in code so it is
  reproducible and reviewable.
- **Keep DOL's original filename.** The pipeline infers fiscal year and quarter
  from it.
- If DOL revises a file, keep both and note the download date — restatements
  happen.

Record for each download: the file name, the page URL, and the date retrieved.
That provenance goes in the write-up under `paper/`.

---

## 4. Fields we care about

Exact names vary by fiscal year; these are the concepts the pipeline needs.

**Targets**

- `CASE_STATUS` — Certified / Denied / Withdrawn / Certified-Expired
  → classification target
- `RECEIVED_DATE` (filing) and `DECISION_DATE` → their difference in days is the
  regression target

**Features (pre-decision only)**

- Employer: name, city, state, and — where present — number of employees, year
  established
- Job: SOC/O*NET code and title, job title, worksite state, education and
  experience requirements
- Wage: offered wage, wage unit, prevailing wage, prevailing wage source
- Filing context: fiscal year, filing quarter, refile indicator, audit-related
  flags where available

**Fairness grouping attributes**

- Employer size (bucketed from employee count, or a proxy such as filing volume)
- Occupation category (SOC major group)
- Geographic region (worksite state → Census region)

**Leakage warning.** Any field that is only knowable at or after adjudication
must stay out of the feature set. `DECISION_DATE` is the clearest example: it
defines the regression target and must never appear as an input.

---

## 5. We develop against synthetic data first

**No DOL download is required to start.** `src/data_prep/make_synthetic.py`
(planned) will generate a synthetic PERM-shaped dataset into
`data/processed/synthetic/` — same column names, same dtypes, same category
levels, plausible date ranges, deliberately imbalanced outcome classes.

Why start synthetic:

- Pipeline, API, app, and tests can be built and CI-tested without shipping or
  waiting on hundreds of MB of government spreadsheets.
- It makes schema assumptions explicit and forces the harmonization layer to be
  written before real data arrives.
- Tests can run anywhere, including on machines that have no `data/raw/`.

**Synthetic data is for plumbing only.** Its correlations are invented. No
accuracy number, SHAP plot, or fairness finding produced from synthetic data is
a result — every reported metric in `reports/` and `paper/` must come from real
DOL files, and anything generated from synthetic input must be labeled as such.
