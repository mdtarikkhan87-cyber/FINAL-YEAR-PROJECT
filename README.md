# Predicting PERM Visa Processing Time & Outcome

An explainable, fairness-audited machine learning system for the U.S. PERM
(Program Electronic Review Management) labor certification process — the first
major step toward an employment-based green card.

---

## 1. Problem Statement

Employers, applicants, and immigration attorneys currently have **no reliable
way to estimate the two things that matter most about a PERM case**:

1. **How long will it take?** Time from filing to final determination ranges
   from a few months to well over a year, driven by audits, backlogs, employer
   characteristics, occupation, and filing period. Public "processing time"
   pages report coarse averages, not case-level estimates.
2. **What will happen?** Cases end as *Certified*, *Denied*, *Withdrawn*, or
   *Certified-Expired*. Applicants often learn the outcome only after months of
   uncertainty — after job offers, relocations, and H-1B extensions have already
   been committed.

The cost is real: employers cannot plan headcount, applicants cannot plan their
lives, and attorneys cannot triage which cases need intervention.

### What this project does

Two complementary predictive tasks over U.S. DOL OFLC PERM disclosure data:

| Task | Type | Target |
|---|---|---|
| **Processing time** | Regression | Days from `RECEIVED_DATE` to `DECISION_DATE` |
| **Case outcome** | Classification | Certified / Denied / Withdrawn / Certified-Expired |

### Why this goes beyond a single accuracy score

One number is not a defensible answer for a system that touches immigration
decisions. The project is built around four additional pillars:

- **Explainability (SHAP)** — global feature importance plus per-case
  explanations: *why* this case is predicted to take 320 days, and which factors
  push the estimate up or down.
- **Fairness audit (fairlearn)** — measure whether error rates and predicted
  outcomes differ systematically across **employer size**, **occupation
  category (SOC)**, and **geographic region**. Report disparity metrics, not
  just aggregate performance.
- **Temporal validation** — train on older filing years, test on newer ones
  (e.g. train FY2018–FY2021, test FY2022–FY2023). Random k-fold splits leak
  future information and overstate real-world performance on a process whose
  backlogs shift year to year.
- **Reproducibility (MLflow)** — every run, parameter set, metric, and artifact
  is tracked, so results in the write-up can be regenerated.

### Explicit non-goals

This is a research and decision-support tool. It does **not** give legal advice,
does **not** claim to predict individual government adjudication decisions with
certainty, and its fairness audit measures *the model's* behavior on historical
data — it is not a statement about DOL's conduct.

---

## 2. Architecture Overview

```
                    ┌──────────────────────────────┐
                    │  DOL OFLC PERM Disclosure    │
                    │  (quarterly Excel / CSV)     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ src/data_prep/                                                 │
 │  ingest → harmonize schemas across years → clean →             │
 │  feature engineering (employer size, SOC group, region,        │
 │  wage ratios, filing quarter, audit flags) → data/processed/   │
 └──────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ src/validation/     temporal split: train on older years,      │
 │                     validate/test on newer years (no leakage)  │
 └──────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ src/models/                                                    │
 │   Regression      : processing-time days  (baseline → XGBoost) │
 │   Classification  : case outcome          (baseline → XGBoost) │
 │   Every run tracked in MLflow                                  │
 └───────────────┬───────────────────────────────┬────────────────┘
                 │                               │
                 ▼                               ▼
 ┌───────────────────────────┐   ┌───────────────────────────────┐
 │ src/explainability/       │   │ src/fairness/                 │
 │  SHAP global + per-case   │   │  fairlearn disparity metrics  │
 │  plots → reports/         │   │  by employer size / SOC /     │
 │                           │   │  region → reports/            │
 └───────────────┬───────────┘   └───────────────┬───────────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
        ┌────────────────────┬───────────────────┬──────────────┐
        │ api/  FastAPI      │ app/  Streamlit   │ paper/       │
        │ /predict/time      │ interactive       │ methods and  │
        │ /predict/outcome   │ what-if explorer  │ findings     │
        │ /explain           │ + SHAP display    │ write-up     │
        └────────────────────┴───────────────────┴──────────────┘
                                 │
                                 ▼
                          Docker (API + app)
```

### Repository layout

```
perm-visa-ml/
├── data/
│   ├── raw/                 # Untouched DOL disclosure files (git-ignored)
│   ├── processed/           # Cleaned, feature-engineered datasets
│   └── README.md            # Where to get the data and which files to grab
├── src/
│   ├── data_prep/           # Ingestion, cleaning, feature engineering
│   ├── models/              # Regression + classification training/inference
│   ├── explainability/      # SHAP explainers and plot generation
│   ├── fairness/            # fairlearn audits across group attributes
│   └── validation/          # Temporal splitting and evaluation protocol
├── api/                     # FastAPI service exposing predictions
├── app/                     # Streamlit front end
├── reports/                 # Generated figures, metrics, audit tables
├── paper/                   # Manuscript, references, final write-up
├── requirements.txt
└── README.md
```

### Design decisions worth knowing

- **Temporal, not random, validation.** PERM backlogs and audit rates shift year
  over year; a random split lets the model see the future.
- **Two targets, one feature pipeline.** Processing time and outcome share
  nearly all inputs, so feature engineering lives in one place and both model
  families consume the same processed tables.
- **Pre-decision features only.** Anything knowable only at or after the
  decision is excluded from the feature set to avoid target leakage.
- **Fairness attributes are structural proxies.** The disclosure data contains
  no applicant race or gender. Audits therefore use employer size, occupation,
  and region, and findings are framed accordingly.

---

## 3. Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data wrangling | pandas, numpy, pyarrow, openpyxl |
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Fairness | fairlearn |
| Experiment tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| UI | Streamlit |
| Visualization | matplotlib, seaborn, plotly |
| Packaging / deploy | Docker |
| Quality | pytest, black, ruff, isort |

---

## 4. Setup Instructions

> **Placeholder — to be finalized once the pipeline exists.** The commands below
> describe the intended workflow; the scripts they reference are not implemented
> yet.

### Prerequisites

- Python 3.10 or newer
- Git
- (Optional) Docker, for the containerized API + app

### Local environment

```bash
python -m venv .venv
```

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

Activate the virtual environment first — `.venv\Scripts\Activate.ps1` on Windows
PowerShell, `source .venv/bin/activate` on macOS/Linux.

### Data

See [data/README.md](data/README.md) for where to download the DOL OFLC PERM
disclosure files, which fiscal years and quarters to collect, and where to put
them. Development begins against **synthetic data**, so no download is required
to run the pipeline for the first time.

### Planned commands (not yet implemented)

```bash
python -m src.data_prep.make_synthetic
```

```bash
python -m src.data_prep.build_dataset
```

```bash
python -m src.models.train --task time
```

```bash
python -m src.models.train --task outcome
```

```bash
python -m src.explainability.run_shap
```

```bash
python -m src.fairness.run_audit
```

```bash
uvicorn api.main:app --reload
```

```bash
streamlit run app/main.py
```

### Docker (planned)

```bash
docker compose up --build
```

---

## 5. Project Status

**Phase 0 — scaffolding.** Folder structure, dependencies, and data
documentation are in place. No models, pipelines, or application code have been
written yet.

- [x] Project scaffold
- [ ] Synthetic data generator
- [ ] Data ingestion + schema harmonization across fiscal years
- [ ] Feature engineering
- [ ] Temporal validation framework
- [ ] Baseline + XGBoost models (regression, classification)
- [ ] SHAP explainability
- [ ] fairlearn fairness audit
- [ ] FastAPI service
- [ ] Streamlit app
- [ ] Dockerization
- [ ] Write-up

---

## 6. Data Source & Attribution

Data originates from the **U.S. Department of Labor, Office of Foreign Labor
Certification (OFLC) Performance Data** disclosure files:
<https://www.dol.gov/agencies/eta/foreign-labor/performance>

These are public records of employer-filed applications. They contain employer
and job identifiers but no applicant names or personally identifying applicant
details. Handle and publish results accordingly.
