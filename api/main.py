"""FastAPI credit-decision service: POST /predict scores one applicant end to end.

Model artifacts (the trained LightGBM model, category encodings, and release metadata) are loaded
once at startup, not per request. A new model takes effect when the service restarts.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.rate_limit import PredictionRateLimiter
from api.schemas import ApplicantRequest, PredictResponse
from api.scoring import InvalidApplicantError, load_artifacts, score_applicant

state: dict = {}
prediction_rate_limiter = PredictionRateLimiter()


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
    version="1.2.0",
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


@app.exception_handler(RequestValidationError)
async def clear_validation_errors(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 body nests each error under loc/msg/type/ctx/url — accurate, but a
    caller has to reconstruct the field name from a list. Flatten it to 'field: message' instead."""
    errors = [f"{'.'.join(str(p) for p in err['loc'][1:])}: {err['msg']}" for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: ApplicantRequest) -> PredictResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return score_applicant(req, state)
    except InvalidApplicantError as exc:
        raise HTTPException(status_code=422, detail=[f"{exc.field}: {exc}"]) from exc
