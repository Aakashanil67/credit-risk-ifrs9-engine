"""FastAPI credit-decision service: POST /predict scores one applicant end to end.

Model artifacts (the trained LightGBM model, its category encodings, and training-set medians
for reason-code phrasing) are loaded once at startup, not per request — retraining or reloading
happens by restarting the service, not on the request path.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import ApplicantRequest, PredictResponse
from api.scoring import load_artifacts, score_applicant

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.update(load_artifacts())
    yield
    state.clear()


app = FastAPI(
    title="Credit Risk & IFRS 9 Engine",
    description="PD scoring, SHAP reason codes, and IFRS 9 ECL for a single loan applicant.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: ApplicantRequest) -> PredictResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return score_applicant(req, state)
