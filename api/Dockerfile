# FastAPI predict service.
#
# models/ is intentionally NOT copied into this image — it's gitignored, and the Kaggle
# competition's terms don't allow redistributing the training data, so there's nothing to bake
# model artifacts from at build time anyway. Mount a host models/ directory at runtime instead
# (see docker-compose.yml), produced by running `python -m src.train_lgbm` once outside Docker.

FROM python:3.12-slim

WORKDIR /app

# libgomp1: LightGBM's compiled wheel links against OpenMP at runtime, and python:slim doesn't
# ship it — the container starts but every predict_proba() call segfaults without this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt pulls in some training-only packages (statsmodels, mlflow, scipy, optbinning)
# this service never calls directly — api/scoring.py imports src.explain, which imports
# src.train_lgbm for build_lgbm_features(), which imports src.baseline for the module-level
# comparison logic, dragging the whole training import graph in at module load time even though
# none of it runs on the request path. Splitting a serving-only import path out of src/ would
# trim the image; not done here since it touches several modules for a size optimisation, not a
# correctness one.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/

EXPOSE 8000

# `python -m uvicorn`, not the bare `uvicorn` console-script entry point: -m adds the current
# directory to sys.path, which is what makes `api.main` resolve as a package. The bare entry
# point's sys.path[0] is the script shim's own directory, not /app, and fails to import api.main.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
