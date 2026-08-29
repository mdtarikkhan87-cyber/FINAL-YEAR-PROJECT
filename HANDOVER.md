# Handover — Predicting PERM Visa Processing Time & Outcome

**Last session:** 2026-08-28
**Phase:** 9 of ~11 — through to Dockerfiles and compose for the full stack.

---

## 1. Where things are

| | |
|---|---|
| Project root | `C:\Users\mdtar\OneDrive\Desktop\final year project\` |
| Python | 3.12.10 (system install — **no virtualenv created yet**) |
| Installed | pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2, xgboost 3.4.1, mlflow 3.15.2, pyarrow 25.0.1, matplotlib 3.10.6. Full `requirements.txt` still not installed (no shap, no fairlearn, no fastapi, no streamlit). |
| Version control | git repo, pushed to https://github.com/mdtarikkhan87-cyber/FINAL-YEAR-PROJECT (**public**). All phases through the Next.js frontend are committed. |

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
│   ├── processed/                     train/test.parquet + metadata.json (GITIGNORED)
│   └── processed_decision/            decision-split copy for the §7.1 comparison (GITIGNORED)
├── src/
│   ├── data_prep/
│   │   ├── make_synthetic.py          synthetic DOL-shaped data generator
│   │   ├── reference.py               state/region maps, wage units, status map, SOC crosswalk
│   │   ├── schema.py                  column alias resolution + validation (raises SchemaError)
│   │   ├── cleaning.py                field-level standardisation (dates, names, SOC, wages, state)
│   │   ├── features.py                target engineering + FEATURE_COLUMNS allow-list
│   │   ├── split.py                   temporal split + cohort-censoring diagnostics
│   │   └── build_dataset.py           orchestrator + CLI
│   ├── models/
│   │   ├── data.py                loads processed data; excludes the split-year feature
│   │   ├── evaluate.py            metrics, always paired with a baseline
│   │   ├── tracking.py            MLflow (sqlite) setup, figures, artifact saving
│   │   ├── train_regressor.py     XGBoost -> processing_days
│   │   ├── train_classifier.py    XGBoost -> outcome
│   │   └── artifacts/             saved models (GITIGNORED)
│   ├── explainability/
│   │   ├── shap_analysis.py       TreeSHAP + additivity check, importance, case selection
│   │   └── run_shap.py            CLI; writes reports/shap_report.md + reports/shap/*.png
│   ├── fairness/
│   │   ├── metrics.py             fairlearn MetricFrames, disparity calcs, flagging
│   │   └── run_audit.py           CLI; writes reports/fairness_report.md + reports/fairness/*.png
│   └── validation/
│       ├── harness.py             fold construction, per-fold fit/score, trend analysis
│       └── run_validation.py      CLI; writes reports/temporal_validation_report.md
├── api/
│   ├── schemas.py                     Pydantic request/response models + validation
│   ├── store.py                       bounded in-memory prediction + SHAP store
│   ├── predictor.py                   model loading, feature derivation, SHAP
│   └── main.py                        FastAPI app (3 endpoints + /health)
├── app/                               empty (Streamlit stub — superseded by web/)
├── docker-compose.yml                 runs api + web together
├── .dockerignore                      root context (api image)
├── web/                               Next.js 16 + Tailwind + recharts frontend
│   ├── app/{page,predict,about}       landing / predictor / methodology
│   ├── components/                    Nav, CaseForm, OutcomeChart, ShapChart, …
│   └── lib/api.ts                     typed client for the FastAPI service
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

## 3b. What was done on 2026-08-28

Built the data preparation pipeline (`src/data_prep/`, 6 modules) and ran it on the
synthetic sample: load → resolve/validate schema → clean → engineer targets → temporal
split → write `data/processed/`.

**The headline finding is in §7.1 below: splitting on filing year produces a censored
test set.** It is the one thing to read before doing anything else with this data.

Then built `src/models/` and trained both XGBoost models. **Both are currently at or
near baseline — see §7.1 and §7.3.** Still no API, no app, no tests.

Installed into system Python (not a venv): `xgboost` 3.4.1, `mlflow` 3.15.2, and
`pyarrow` 25.0.1 (an mlflow dependency). Because pyarrow now exists,
`build_dataset.py --format auto` writes **parquet** rather than CSV.

---

## 4. How to run it

Regenerate the synthetic dataset:

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

### Data preparation pipeline

```bash
python -m src.data_prep.build_dataset
```

Reads every CSV/Excel file in `data/raw/`, resolves their columns onto the canonical
schema, cleans, engineers targets, splits chronologically, and writes
`data/processed/{train,test}.{parquet|csv}` plus `metadata.json` (provenance, row
accounting, split definition, cohort diagnostics).

| Flag | Effect |
|---|---|
| `--split-on {filing,decision}` | which fiscal year keys the split (default `filing` — **see §7.1**) |
| `--test-years N` | how many of the newest cohorts form the test set (default 1) |
| `--drop-incomplete-cohorts` | exclude cohorts whose observation window is censored |
| `--format {auto,parquet,csv}` | `auto` uses parquet when pyarrow is installed, else CSV |
| `--raw-dir` / `--out-dir` / `--pattern` | override input/output locations |
| `--no-preview` | skip the printed report |

A file whose columns cannot be resolved fails with `SchemaError` (exit 2), naming each
missing column, its accepted raw spellings, and the closest column actually present. Add
new spellings to `COLUMN_ALIASES` in `schema.py` — never to the cleaning code.

### Model training

```bash
python -m src.models.train_regressor
```

```bash
python -m src.models.train_classifier
```

| Flag | Effect |
|---|---|
| `--data-dir` | which processed dataset to use (default `data/processed`) |
| `--val-strategy {random,temporal}` | early-stopping hold-out (default `random` — **see §7.3**) |
| `--keep-year-feature` | keep the split-year column as a feature (**breaks the model** — §7.3) |
| `--merge-expired` | classifier only: fold `certified_expired` into `certified` (3 classes) |
| `--balanced` | classifier only: balanced sample weights; macro-F1 up, accuracy down |
| `--log-target` | regressor only: train on `log1p(days)`; metrics stay in days |
| `--n-estimators` / `--learning-rate` / `--max-depth` / `--early-stopping-rounds` | XGBoost tuning |

Models save to `src/models/artifacts/` as a native XGBoost `.json` (portable) plus a
`_bundle.joblib` carrying the fitted estimator, feature order, and category levels —
the booster alone cannot reproduce a prediction without those.

### SHAP explainability

```bash
python -m src.explainability.run_shap
```

Explains whichever models sit in `src/models/artifacts/`, on the test set from
`--data-dir`. Writes `reports/shap_report.md` and 13 PNGs to `reports/shap/`.
`--max-rows` caps rows explained (default 3000; TreeSHAP is exact regardless).

Both models must come from the **same** `--data-dir` — the artifacts directory holds one
bundle per task, so training the regressor on one split and the classifier on another
leaves an incoherent pair and the report will silently compare unlike things.

### Fairness audit

```bash
python -m src.fairness.run_audit
```

Audits both saved models across employer size, SOC major group, and worksite region.
Writes `reports/fairness_report.md` plus 2 PNGs to `reports/fairness/`.
`--min-group-size` (default 30) sets the size below which groups are shown but never
flagged.

### Temporal validation harness

```bash
python -m src.validation.run_validation
```

Re-fits both models once per year boundary and writes
`reports/temporal_validation_report.md`, two PNGs to `reports/validation/`, and two CSVs
of the per-fold metrics.

| Flag | Effect |
|---|---|
| `--scheme {expanding,rolling}` | growing vs fixed-length training window (default expanding) |
| `--window N` | training years per fold when rolling (default 3) |
| `--min-train-years N` | minimum training years before the first fold (default 2) |
| `--keep-year-features` | keep absolute-year columns — **worse in 4 of 5 folds**, see §7.6 |
| `--balanced` | balanced class weights in the classifier folds |

### API

```bash
uvicorn api.main:app --reload
```

Then http://127.0.0.1:8000/docs for interactive Swagger UI.

| Endpoint | Purpose |
|---|---|
| `POST /predict/processing-time` | days from filing to decision, + estimated decision date |
| `POST /predict/outcome` | predicted class + probability for all four outcomes |
| `GET /explain/{prediction_id}` | SHAP attribution for a stored prediction |
| `GET /health` | model + store status |

The request is a **case**, not a feature vector — `soc_major_group`, `wage_ratio`,
`worksite_region` and the rest are derived server-side by importing the same
`src/data_prep` functions the training pipeline used. Re-implementing them in the API
would create training/serving skew that fails silently.

Predictions and their SHAP values are held in a bounded in-memory store (500 most recent,
oldest evicted, cleared on restart). `/explain` returns the attribution for the exact row
that produced the answer rather than recomputing it. Every prediction response carries a
`model_quality` block stating the model's test metric, its baseline, and its known
weaknesses — the outcome model's caveat says outright that it does not beat its baseline.

### Docker

```bash
docker compose up --build
```

Then http://localhost:3000 (site) and http://localhost:8000/docs (API). Override ports
or the API location with `WEB_PORT`, `API_PORT`, `API_INTERNAL_URL`.

**Never built — see §7.9.** Run the training pipeline first so the model bundles exist.

### MLflow

```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

Run from the project root, then open http://127.0.0.1:5000. Experiments:
`perm-processing-time` and `perm-outcome`.

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

### 7.1 Which date does the temporal split key on? (MOST IMPORTANT)

Measured on the synthetic sample, 50,000 rows, one test cohort:

| `--split-on` | test rows | test % | test mean days | test max days | verdict |
|---|---|---|---|---|---|
| `filing` (current default) | 1,101 | 2.2% | 241 | **356** | censored — unusable |
| `filing --drop-incomplete-cohorts` | 7,219 | 18.3% | 407 | 980 | valid, costs 4 cohorts |
| `decision` | 7,000 | 14.0% | 371 | 1,060 | valid, keeps everything |

A disclosure extract contains only *decided* cases. Cases filed in the newest year that
are still pending are simply absent, so the newest **filing** cohort holds only the fast
ones: its maximum observed processing time is 356 days when the training set's 90th
percentile is 577. The same bias hits the outcome target — denied cases take longer, so
the censored test set shows 2.4% denials against 5.6% in training. A processing-time
model scored on that test set would look excellent and mean nothing.

`--split-on decision` has no censored cohorts by construction: the file is *grouped* by
decision date, so every decision in the year is present. Its cost is that some training
cases were filed after some test cases.

**Recommendation: `--split-on decision`.** The current default is `filing` because that
is what was specified; change the default in `build_dataset.py` once you have decided.
Whichever you pick, state it and the reasoning in `paper/` — this is exactly the kind of
methodological choice the "temporal validation" pillar is supposed to demonstrate.

### 7.3 Both models are currently at or near baseline

| Model | split | metric | model | baseline |
|---|---|---|---|---|
| Regressor | filing | RMSE (days) | 124.6 | **68.5** (worse) |
| Regressor | decision | RMSE (days) | **155.8** | 182.5 (+14.7%) |
| Classifier | filing | ROC AUC (OvR) | 0.498 | chance |
| Classifier | decision | ROC AUC (OvR) | 0.523 | chance |

Three findings, in order of importance:

1. **The split-year column cannot be a feature.** `filing_fiscal_year` was the
   highest-gain feature (38.9%) during fitting, then took an unseen value (2024) at
   test time. Trees cannot extrapolate past their fitted range, so it silently became a
   constant and the model scored *worse than predicting the training median*.
   `data.load_modelling_data(drop_split_year=True)` now excludes it by default;
   `--keep-year-feature` restores the broken behaviour for comparison. Seasonality still
   reaches the model through `filing_fiscal_quarter` and `filing_month`, which repeat.

2. **Early stopping needs a random within-train hold-out, not a temporal one.** With a
   temporal hold-out, the last training cohort sits in a different backlog regime, so
   every additional tree looks worse and early stopping selects round 0 — a one-tree
   model. Switching to a random hold-out drawn from inside the training period (leaking
   nothing; the test set is untouched) took the regressor from `best_iteration=0` to 164.
   Controlled by `--val-strategy {random,temporal}`, default `random`.

3. **Outcome is barely predictable from the observable features, and that is correct.**
   The generator's dominant denial driver is the *latent* audit variable, which is
   deliberately not a column because real disclosure files do not have one. Per-class
   OvR AUC on the decision split: denied 0.574, withdrawn 0.521, certified_expired 0.473.
   The `denied` probability never exceeds 0.36 so it is never the argmax, which is why
   the confusion matrix is degenerate. `--balanced` makes rare classes predictable at all
   (macro-F1 0.233 -> 0.245) at the cost of accuracy (0.871 -> 0.842), but recall stays
   near zero. Do not tune this away — it is a property of the data, not a bug.

**The principled next step for the regressor** is a *recent-backlog* feature: the median
processing time of cases **decided in the months before this case was filed**. That is
knowable at filing time, uses only past decisions, and proxies the regime the year column
was illegitimately standing in for. It is the single change most likely to move the
regression metrics.

### 7.4 SHAP found a second out-of-range year feature

`drop_split_year` only excludes the column the split was *keyed* on. On the decision
split that is `decision_fiscal_year`, so **`filing_fiscal_year` stayed in the feature set
and took 37.5% of the regressor's SHAP attribution.** Its training range is 2016-2023, but
15.7% of test rows are filed in FY2024. Splitting the test metric on that boundary:

| test subset | n | model RMSE | baseline RMSE |
|---|---:|---:|---:|
| filing year in range | 5,899 | 167.4 | 197.2 (beats baseline) |
| filing year out of range | 1,101 | 64.8 | 58.8 (**worse than baseline**) |

The headline +14.7% RMSE averages these together and hides the second row. The fix is to
exclude *every* absolute-year column under a temporal split, not just the split key —
or better, replace them with the recent-backlog feature from §7.3. `run_shap.py` now
reports this automatically for whichever feature tops the attribution.

### 7.5 The fairness audit cannot clear the classifier yet

The Phase 4 classifier predicts `certified` for all 7,000 test cases, so demographic
parity difference and equalized odds difference are **exactly 0.000 for every grouping
attribute**. That is arithmetic, not evidence: a model that certifies everyone treats all
groups identically and is useless. `run_audit.py` detects this and refuses to report it
as a pass. **Never quote those zeros.**

Retraining with `--balanced` gives a genuinely non-degenerate audit, verified:

| attribute | DP difference | EO difference | selection-rate ratio |
|---|---:|---:|---:|
| employer size | 0.023 | 0.042 | 0.977 |
| occupation (SOC) | 0.052 | 0.052 | 0.948 |
| worksite region | 0.037 | 0.039 | 0.963 |

All under the 0.10 threshold and well above the four-fifths screen — a real pass. The
audit in `reports/` is the unbalanced Phase 4 model, because that is what the artifacts
directory holds; regenerate with the balanced model before citing outcome fairness.

**Regressor:** zero flags, but only because the flagging is *relative*. The model
under-predicts by 89 days for everyone; group departures from that global bias are all
under 21 days. An earlier version flagged absolute bias and produced 22 near-identical
"disparities" that were really one model defect repeated. The global optimism is an
accuracy problem (§7.3), not a fairness one, and no fairness intervention would fix it.

### 7.6 Temporal validation: the model is stable, not degrading

Five expanding-window folds (train ≤ N, test N+1) over `data/processed_decision`:

| fold | train | test | RMSE | baseline | improvement |
|---:|---|---:|---:|---:|---:|
| 1 | 2018-2019 | 2020 | 145.9 | 167.6 | +12.9% |
| 2 | 2018-2020 | 2021 | 162.0 | 191.6 | +15.5% |
| 3 | 2018-2021 | 2022 | 211.7 | 250.4 | +15.5% |
| 4 | 2018-2022 | 2023 | 229.3 | 272.6 | +15.9% |
| 5 | 2018-2023 | 2024 | 159.8 | 182.5 | +12.5% |

**Raw RMSE nearly doubles then falls back — but that is the target moving, not the model
failing.** The baseline moves in lockstep (168 → 273 → 183). Improvement over baseline is
flat at −0.05 pp/year. Quoting raw RMSE across periods of differing difficulty would
manufacture a trend that is not there; the improvement column is the honest read.

The classifier is flat and near-chance in every fold (ROC AUC 0.530–0.541, accuracy never
more than 0.0001 from the majority baseline, 1–2 classes ever predicted). Stable, not
drifting — a feature-availability problem, not a drift problem.

**The harness confirms §7.4 across all five folds.** Running `--keep-year-features`:

| fold | dropped (default) | kept | baseline |
|---:|---:|---:|---:|
| 1 | **145.9** | 170.8 | 167.6 |
| 2 | **162.0** | 187.3 | 191.6 |
| 3 | **211.7** | 232.6 | 250.4 |
| 4 | **229.3** | 235.4 | 272.6 |
| 5 | **159.8** | 155.8 | 182.5 |

Dropping the absolute-year column wins in 4 of 5 folds, by 6–25 days; in fold 1 keeping it
makes the model *worse than baseline*. This is far stronger evidence than the single-split
finding, and it settles the question: absolute-year columns do not belong in a temporally
validated model.

`--scheme rolling --window 3` also helps during the backlog shift (fold 3: 204.2 vs 211.7;
fold 4: 206.9 vs 229.3) — forgetting older regimes is worth testing properly.

### 7.7 The API: omitted optional fields are not neutral

Same case, sparse vs complete input:

| request | predicted days |
|---|---:|
| required fields only (+ employees, refile) | **468.2** |
| same case with citizenship, class_of_admission, education, skill level, NAICS, year established, experience | **263.6** |

A 205-day swing. The cause: `citizenship` was never missing in the training data, so a
missing value is a branch the model never really fitted — SHAP attributed **+134 days
(35% of the total attribution)** to `citizenship = missing` alone. The API now computes
this per request and returns an `input_warnings` list naming any absent field absorbing
more than 10% of the attribution. **Do not treat the optional fields as genuinely
optional.**

A cleaner fix for later: train with realistic missingness in those columns so the missing
branch is actually fitted, or require the fields outright.

### 7.8 Frontend notes

**Run both servers.** The web app is useless without the API:

```bash
uvicorn api.main:app --port 8000
```
```bash
cd web && npm run dev
```

**CORS** is enabled in `api/main.py` for `localhost`/`127.0.0.1` on ports 3000-3001.
Override with `PERM_API_CORS_ORIGINS` (comma-separated). Verified: preflight from
`http://localhost:3000` returns the correct `access-control-allow-origin`.

**Form options come from the model, not a hardcoded list.** `GET /meta/options` returns
the fitted bundle's own category levels (37 SOC codes, 26 states, 30 citizenships…), so
the UI cannot offer a value the model has never seen. A hand-maintained list would drift
out of sync with the next retrain.

**Next 16, not 14 or 15.** npm audit flagged a high-severity advisory covering all of
14.x and 15.x; only 16.3.3 is patched. Current install: 0 vulnerabilities.

**The dev server does not render inside sandboxed browser tooling.** Turbopack's dev
chunk URLs get 403'd there, so hydration never starts and the page looks inert. `npm run
build && npx next start` works fine. This affects automated browser testing only — a
normal browser on localhost is unaffected.

**`agentRules: false`** is set in `next.config.mjs`; Next 16 otherwise writes its own
`AGENTS.md`/`CLAUDE.md` into `web/` on every dev start.

### 7.9 Docker: written, not built

`api/Dockerfile`, `web/Dockerfile`, and `docker-compose.yml` exist and are internally
consistent, but **no image has ever been built**. Docker Desktop is not installed here
(WSL2 is present; Docker is not). Everything short of the build was verified:

- compose YAML parses; every `COPY` source path exists
- `.dockerignore` keeps the `*_bundle.joblib` model files and drops the two unused
  ~2 MB native `*_xgb.json` exports
- the API's serving dependency set was derived by importing `api.main` and listing the
  third-party modules actually pulled in
- **the runtime wiring was tested for real** by running `.next/standalone/server.js`
  with `API_INTERNAL_URL` set and `NEXT_PUBLIC_API_BASE` unset — exactly the container
  configuration. Full flow passed; the browser made only same-origin `/api/*` calls.

First `docker compose up --build` may still surface image-level problems — a missing
system library, a base-image tag change, an amd64/arm64 wheel gap. Expect to iterate once.

**The API-URL design is the part worth understanding.** `NEXT_PUBLIC_*` variables are
inlined into the client bundle at *build* time. Baking `http://api:8000` would ship a
hostname only resolvable inside the Docker network and every browser request would fail;
baking `http://localhost:8000` would work locally and break on deployment, requiring an
image rebuild to change a URL. So the browser always calls same-origin `/api/*`, and
`web/app/api/[...path]/route.ts` forwards to `API_INTERNAL_URL` read **per request**.
One variable, no rebuild, correct in both places — and CORS drops out of the browser
path entirely (the API still sets it for direct callers such as `/docs`).

`npm run dev` is unchanged: `web/.env.local` sets `NEXT_PUBLIC_API_BASE`, so the browser
calls the API directly and exercises CORS.

**Prerequisite:** the API image copies `src/models/artifacts/*.joblib`, which are
gitignored. Generate them before the first build, or the api container starts, reports
itself unhealthy, and returns 503 (and compose will hold `web` back, by design).

### 7.2 Remaining questions

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

- [x] `git init` + first commit — pushed to
      https://github.com/mdtarikkhan87-cyber/FINAL-YEAR-PROJECT (public)
- [x] **`src/data_prep/build_dataset.py`** — schema resolution/validation, cleaning,
      target engineering, temporal split, `data/processed/` output
- [x] Feature engineering — employer size buckets, SOC major group, Census region,
      annualised wage ratio, filing quarter (31 features, `ft.FEATURE_COLUMNS`)
- [x] Baseline + XGBoost models for both targets, MLflow tracking (`src/models/`)
- [x] SHAP explainability (`src/explainability/`) — 13 plots + `reports/shap_report.md`
- [x] Fairness audit (`src/fairness/`) — `reports/fairness_report.md` + 2 plots
- [x] Temporal validation harness (`src/validation/`) — 5 folds, expanding + rolling
- [x] FastAPI service (`api/`) — predict/explain endpoints, verified end to end
- [x] Next.js frontend (`web/`) — landing, predictor, methodology; full flow verified
- [x] Dockerfiles + compose — **written and validated, but never built: Docker is
      not installed on this machine (see §7.9)**
- [ ] **Decide §7.1** (split key) and make it the default in `build_dataset.py`
- [ ] Add the recent-backlog feature described in §7.3 — the highest-value modelling change
- [ ] Create a real `.venv` and `pip install -r requirements.txt`; deps currently sit in
      system Python
- [ ] Tests for `src/data_prep/` and `src/models/` — there are none
- [ ] SHAP, fairlearn audit, FastAPI, Streamlit, Docker, write-up

Suggested next task: settle §7.1, then the recent-backlog feature. SHAP is ready to start
whenever, since the saved bundles carry the fitted model plus its category levels.

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
- **MLflow 3.x refuses the old `./mlruns` file store.** Tracking runs on SQLite at
  `mlflow.db` with artifacts in `mlartifacts/` (both gitignored). Launch the UI with
  `mlflow ui --backend-store-uri "sqlite:///mlflow.db"` from the project root — plain
  `mlflow ui` will not find the runs.
- `data/processed_decision/` holds a decision-split copy of the dataset for the §7.1
  comparison. It is gitignored and rebuildable with
  `python -m src.data_prep.build_dataset --split-on decision --out-dir data/processed_decision`.
