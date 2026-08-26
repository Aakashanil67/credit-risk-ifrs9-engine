# FastAPI predict service.
#
# models/*.joblib are committed to git (learned parameters, not the Kaggle dataset, so this stays
# clear of the redistribution terms — see DECISIONS.md) and COPYed in below, so the image is
# self-sufficient. docker-compose.yml also volume-mounts ./models over this path at dev time so a
# retrain doesn't need a rebuild. A platform that builds this Dockerfile directly, with no host
# filesystem to mount from, still gets a working image. That gap is what broke the first Render
# deploy: the mount was doing all the work and nobody noticed.

FROM python:3.12-slim

WORKDIR /app

# libgomp1: LightGBM's compiled wheel links against OpenMP at runtime, and python:slim doesn't
# ship it — the container starts but every predict_proba() call segfaults without this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The API has its own pinned runtime set. Installing the scorecard, MLflow and notebook tooling
# here added several minutes to builds without contributing to a prediction request.
COPY requirements/api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/

EXPOSE 8000

# `python -m uvicorn`, not the bare `uvicorn` console-script entry point: -m adds the current
# directory to sys.path, which is what makes `api.main` resolve as a package. The bare entry
# point's sys.path[0] is the script shim's own directory, not /app, and fails to import api.main.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
