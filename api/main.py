"""FastAPI credit-decision service: POST /predict scores one applicant end to end.

Model artifacts (the trained LightGBM model, category encodings, and release metadata) are loaded
once at startup, not per request. A new model takes effect when the service restarts.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.observability import request_log_record
from api.rate_limit import PredictionRateLimiter
from api.schemas import ApplicantRequest, PredictResponse
from api.scoring import InvalidApplicantError, load_artifacts, score_applicant
from src.config import SERVICE_VERSION

state: dict = {}
prediction_rate_limiter = PredictionRateLimiter()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.update(load_artifacts())
    yield
    state.clear()


app = FastAPI(
    title="Credit Risk & IFRS 9 Engine",
    description=(
        "Public-demo PD scoring, SHAP reason codes, and an illustrative 12-month loss estimate "
        "for one loan applicant."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def limit_prediction_requests(request: Request, call_next):
    if request.url.path == "/predict":
        client_id = request.client.host if request.client else "unknown"
        if not prediction_rate_limiter.allow(client_id):
            return JSONResponse(
                status_code=429,
                content={"detail": "prediction limit reached; try again in one minute"},
            )
    return await call_next(request)


@app.middleware("http")
async def log_request_metadata(request: Request, call_next):
    """Log only fixed operational fields after every request, including rate-limit responses."""
    started_at = time.perf_counter()
    request_id = uuid4().hex
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        record = request_log_record(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=(time.perf_counter() - started_at) * 1_000,
            service_version=SERVICE_VERSION,
            model_version=state.get("metadata", {}).get("model_version"),
        )
        logger.info(json.dumps(record, sort_keys=True))


@app.exception_handler(RequestValidationError)
async def clear_validation_errors(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 body nests each error under loc/msg/type/ctx/url — accurate, but a
    caller has to reconstruct the field name from a list. Flatten it to 'field: message' instead."""
    errors = [f"{'.'.join(str(p) for p in err['loc'][1:])}: {err['msg']}" for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "service_version": SERVICE_VERSION,
        "model_version": state.get("metadata", {}).get("model_version"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: ApplicantRequest) -> PredictResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return score_applicant(req, state)
    except InvalidApplicantError as exc:
        raise HTTPException(status_code=422, detail=[f"{exc.field}: {exc}"]) from exc
