# FastAPI backend: PERM processing-time and outcome prediction.
#
#   docker build -f api/Dockerfile -t perm-api .
#
# NOTE: the build context is the PROJECT ROOT, not api/. The service imports
# src/data_prep and src/explainability, so both api/ and src/ must be inside the
# context. Build from the repository root or use the compose file.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp is required by XGBoost (OpenMP runtime) and is not in python:slim.
# Without it, importing xgboost fails at container start with a missing
# libgomp.so.1, which is an unhelpful error to debug from a crash loop.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so edits to application code do not invalidate the
# (slow, multi-hundred-megabyte) pip layer.
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Application code. src/ carries the cleaning and feature-derivation functions
# the API reuses so that serving matches training exactly, plus the saved model
# bundles under src/models/artifacts/.
COPY src/ ./src/
COPY api/ ./api/

# Run unprivileged.
RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV PERM_API_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"

# Reports unhealthy until the model bundles are actually loaded, not merely
# until the process is listening. A container that answers HTTP but cannot
# predict is not ready for the frontend to depend on.
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD python -c "import json,urllib.request,sys; \
d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); \
sys.exit(0 if d.get('models_loaded') else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
