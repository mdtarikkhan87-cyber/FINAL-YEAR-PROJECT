# Deployment

Frontend on **Vercel**, backend on **Railway** or **Render**. Roughly 20 minutes
end to end, all on free tiers.

Nothing here has been deployed yet. The configuration is verified locally — the
backend runs on an arbitrary `PORT` with production CORS, and the frontend
reaches it purely through an environment variable — but the platform steps below
are written from their documented behaviour, not from a completed deploy. Expect
one or two rounds of iteration.

---

## 0. Why the backend cannot go on Vercel

Worth stating up front, because it is the obvious first instinct.

The API loads XGBoost, SHAP, scikit-learn, pandas and SciPy into memory. That
dependency stack is very likely to exceed Vercel's serverless function size
limit, and even if it squeezed in, every cold start would re-unpickle both model
bundles and rebuild two SHAP explainers. It is the wrong execution model for
this workload. Railway and Render run an ordinary long-lived container, which is
what it needs.

There is one further constraint that matters more than size:

> **The prediction store is in-process memory.** `POST /predict` returns a
> `prediction_id`, and `GET /explain/{id}` looks it up in a dictionary held by
> that Python process. On a single always-on container this is correct. Scale to
> two replicas and roughly half of all `/explain` calls will 404, because the
> request lands on the instance that did not make the prediction.
>
> **Deploy the backend with exactly one instance until this is moved to Redis or
> a database.** Both platforms default to one; do not raise it.

---

## 1. Prerequisites

- The repository pushed to GitHub (already done).
- Accounts on [vercel.com](https://vercel.com) and either
  [railway.app](https://railway.app) or [render.com](https://render.com).
- The trained model bundles present in the repo — see §2.

---

## 2. Model artifacts

**The bundles are committed.** `src/models/artifacts/*_bundle.joblib` totals
about 3.9 MB, small enough that object storage would be pointless ceremony.
Clone the repo and the deployed backend has its models; no upload step, no
external bucket, no credentials.

| File | Size | Committed |
|---|---|---|
| `processing_time_xgb_bundle.joblib` | 1.7 MB | yes |
| `outcome_xgb_bundle.joblib` | 2.2 MB | yes |
| `*_metadata.json` | ~12 KB | yes |
| `*_xgb.json` (native XGBoost export) | ~4.3 MB | **no** — nothing loads them |

The native exports are excluded deliberately: they hold the raw booster without
the category levels and feature order the service needs to build a prediction
row, so the API never reads them.

### Regenerating them

Required after changing features, retraining, or upgrading xgboost/scikit-learn:

```bash
python -m src.data_prep.make_synthetic
```
```bash
python -m src.data_prep.build_dataset --split-on decision --out-dir data/processed_decision
```
```bash
python -m src.models.train_regressor --data-dir data/processed_decision
```
```bash
python -m src.models.train_classifier --data-dir data/processed_decision
```

Then commit the changed `*_bundle.joblib` files. The whole run takes a couple of
minutes.

### Two things to know about these files

**They are version-coupled.** A joblib bundle pickled with xgboost 3.4.1 /
scikit-learn 1.7.2 / numpy 2.2.6 must be unpickled by compatible versions.
`api/requirements.txt` pins that exact set, which is why it is pinned rather than
ranged. If you bump those pins, retrain and recommit the bundles in the same
change, or the container will fail at startup with an unpickling error.

**They are pickles, and pickles execute code on load.** Loading your own
artifacts from your own repository is fine. Never point
`PERM_API_ARTIFACT_DIR` at a bundle from a source you do not control.

### If you would rather not commit them

Set `PERM_API_ARTIFACT_DIR` to a mounted volume and copy the bundles there —
Railway volumes and Render disks both work. The service starts either way: with
no artifacts it stays up, reports `degraded` on `/health`, returns 503 from
`/ready` and from both prediction endpoints, and logs exactly which file it
could not find.

---

## 3. Backend → Railway

1. **New Project → Deploy from GitHub repo**, pick the repository.
2. **Settings → Build**: set **Dockerfile Path** to `api/Dockerfile` and
   **Root Directory** to `/` (empty). The build context must be the repository
   root — the API imports `src/`, so a context of `api/` will fail.
3. **Variables** — add these:

   | Variable | Value |
   |---|---|
   | `PERM_API_ENV` | `production` |
   | `PERM_API_CORS_ORIGINS` | *(leave unset for now; §5)* |

   Do **not** set `PORT`. Railway injects it, and the container's
   `CMD uvicorn ... --port ${PORT:-8000}` picks it up.
4. **Settings → Networking → Generate Domain.** Note the URL, e.g.
   `https://perm-api-production.up.railway.app`.
5. **Settings → Health Check**: path `/health`, timeout 300s.

   Use `/health`, not `/ready`. `/health` is liveness and stays 200 even when the
   models are missing; `/ready` returns 503 in that case and would put the
   service into a restart loop instead of leaving it up to tell you what is
   wrong. Check `/ready` yourself after deploying.

Confirm:

```bash
curl https://YOUR-API.up.railway.app/ready
```

Expect `{"ready": true, "models_loaded": ["regressor", "classifier"], ...}`.

---

## 3b. Backend → Render (alternative)

1. **New → Web Service**, connect the repository.
2. **Runtime: Docker**. **Dockerfile Path** `api/Dockerfile`, **Docker Build
   Context Directory** `.` (the repository root — same reason as above).
3. **Instance Type**: Free works. **Do not enable autoscaling** (see the
   single-instance note in §0).
4. **Environment**: `PERM_API_ENV=production`. Again, do not set `PORT`.
5. **Health Check Path**: `/health`.

Render's free tier spins the service down after inactivity, so the first request
after an idle period takes 30–60 seconds while the container restarts and
re-loads the models. The frontend will show "API offline" during that window.
Railway does not idle in the same way.

---

## 4. Frontend → Vercel

1. **Add New → Project**, import the repository.
2. **Root Directory: `web`.** This is the one setting people miss — the Next app
   is not at the repository root, and Vercel will fail to detect a framework
   without it.
3. Framework preset **Next.js**; leave build and output settings default.
4. **Environment Variables** — add one:

   | Variable | Value | Environments |
   |---|---|---|
   | `API_INTERNAL_URL` | `https://YOUR-API.up.railway.app` | Production, Preview, Development |

   No trailing slash.

   **Do not set `NEXT_PUBLIC_API_BASE`.** See §6 for why this distinction is the
   crux of the whole setup.
5. **Deploy.** Note the URL, e.g. `https://perm-predictor.vercel.app`.

---

## 5. Point the two at each other

The frontend already knows the backend from step 4. The remaining question is
whether the backend needs to know the frontend, and that depends on which of two
arrangements you want.

### Recommended: proxy mode (no CORS at all)

The browser calls `/api/*` on the Vercel domain; Vercel's Next server forwards
server-side to `API_INTERNAL_URL`. The browser never makes a cross-origin
request, so CORS never enters the picture.

- Vercel: `API_INTERNAL_URL=https://YOUR-API.up.railway.app`
- Railway/Render: leave `PERM_API_CORS_ORIGINS` **unset**

The backend logs a note that no origins are configured. That is expected and is
the more secure arrangement — no browser origin is granted access to the API at
all.

### Alternative: direct mode (browser calls the API)

Skips the proxy hop, at the cost of needing CORS and exposing the API URL in the
client bundle.

- Vercel: set `NEXT_PUBLIC_API_BASE=https://YOUR-API.up.railway.app`
  (**requires a redeploy to change** — it is inlined at build time)
- Railway/Render: `PERM_API_CORS_ORIGINS=https://perm-predictor.vercel.app`

Vercel preview deployments get their own URLs (`*-git-branch-you.vercel.app`),
so in direct mode every preview needs adding to `PERM_API_CORS_ORIGINS` or its
requests will be blocked. Proxy mode has no such problem. This alone is a good
reason to prefer it.

Either way, verify from the deployed site rather than from `curl`: open the
Vercel URL, go to **Predict**, and check the header shows **API online**.

---

## 6. Why `API_INTERNAL_URL` and not `NEXT_PUBLIC_API_BASE`

This is the single design decision most worth understanding, because getting it
wrong produces a site that works locally and fails in production.

`NEXT_PUBLIC_*` variables are **inlined into the JavaScript bundle at build
time**. They are baked into the artifact and cannot be changed without a
rebuild. That makes them wrong for a URL that differs between environments:

- Bake `http://api:8000` (the Docker Compose hostname) and the visitor's browser
  cannot resolve it — every request fails.
- Bake `http://localhost:8000` and it works on your laptop and breaks the moment
  it is deployed.
- Bake the production URL and every preview and local run points at production.

`API_INTERNAL_URL` is read **server-side, per request**, by
`web/app/api/[...path]/route.ts`. Change it in the Vercel dashboard and the next
request uses the new value — no rebuild. The browser only ever calls its own
origin.

```
browser ──/api/predict──▶ Vercel (Next server) ──API_INTERNAL_URL──▶ Railway (FastAPI)
          same origin                            read per request
```

---

## 7. Environment variable reference

### Backend (Railway / Render)

| Variable | Default | Purpose |
|---|---|---|
| `PERM_API_ENV` | `development` | `production` disables the localhost CORS fallback and refuses a `*` origin |
| `PERM_API_CORS_ORIGINS` | dev: localhost<br>prod: none | Comma-separated exact origins, scheme included. Unset in proxy mode |
| `PERM_API_ARTIFACT_DIR` | `src/models/artifacts` | Where the `*_bundle.joblib` files live |
| `PERM_API_STORE_CAPACITY` | `500` | Predictions retained in memory for `/explain` |
| `PERM_API_ENABLE_DOCS` | `true` | `false` hides `/docs`, `/redoc`, `/openapi.json` |
| `PORT` | `8000` | **Injected by the platform. Do not set it manually.** |

There are no secrets. No database, no third-party keys, no signing material —
the service loads two files and answers HTTP. `api/config.py` is where one would
go if that changes.

### Frontend (Vercel)

| Variable | Default | Purpose |
|---|---|---|
| `API_INTERNAL_URL` | `http://127.0.0.1:8000` | Backend URL, read per request. **Set this.** |
| `NEXT_PUBLIC_API_BASE` | *(unset)* | Only for direct mode. Build-time inlined; leave unset |

---

## 8. Verifying a deployment

```bash
curl https://YOUR-API.up.railway.app/
```
Service name, version, environment, and the endpoint list.

```bash
curl https://YOUR-API.up.railway.app/ready
```
`200` with `"ready": true`. A `503` means the models did not load — check the
platform logs for `model load failed`, which names the missing file.

```bash
curl -X POST https://YOUR-API.up.railway.app/predict/processing-time -H "Content-Type: application/json" -d '{"filing_date":"2024-03-15","soc_code":"15-1252","worksite_state":"CA","prevailing_wage":145000,"offered_wage":162000}'
```

Then open the Vercel URL and run a prediction from the Predict page. If the
header says **API offline**, the frontend cannot reach the backend: check
`API_INTERNAL_URL` for a typo or trailing slash, and confirm `/ready` returns 200.

---

## 9. Troubleshooting

| Symptom | Cause |
|---|---|
| Vercel build fails, no framework detected | **Root Directory** not set to `web` |
| Backend build fails on `COPY src/` | Build context set to `api/` instead of the repository root |
| Container starts then the platform kills it | Health check pointed at `/ready` instead of `/health`, or `PORT` set manually |
| `/ready` returns 503, logs show `model bundle not found` | Bundles absent — see §2 |
| Site shows "API offline" | `API_INTERNAL_URL` wrong, or free-tier backend cold-starting (wait 60s) |
| Browser console shows a CORS error | Direct mode without the Vercel origin in `PERM_API_CORS_ORIGINS`. Switch to proxy mode |
| `ConfigError` at startup | Deliberate. The message names the variable and the fix |
| `/explain/{id}` returns 404 intermittently | More than one backend instance — scale to 1 (§0) |
| Container exits: `libgomp.so.1: cannot open shared object file` | Base image changed; `libgomp1` must stay in the `apt-get install` line |

---

## 10. Before sharing the link

The site labels itself as synthetic in three places — a banner on every page, a
`model_quality` block on every prediction, and a dedicated Methodology page. Keep
all three. The models are fitted to invented data and the outcome classifier does
not beat its own baseline; a stranger arriving at the URL needs to be able to see
that without reading the repository.
