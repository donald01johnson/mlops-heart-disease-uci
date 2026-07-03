import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
import joblib
import pandas as pd

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
    )

from src.data_pipeline import numeric_features, categorical_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Heart Disease Risk API")

MODEL_PATH = "models/final_model_rf.pkl"
model = joblib.load(MODEL_PATH)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["status"],  # "success", "error"
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = perf_counter()
    try:
        response = await call_next(request)
        duration = perf_counter() - start

        endpoint = request.url.path
        method = request.method
        status = response.status_code

        REQUEST_COUNT.labels(method=method,
                             endpoint=endpoint,
                             http_status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

        logger.info(
            "path=%s method=%s status=%s duration_ms=%.2f",
            endpoint,
            method,
            status,
            duration * 1000.0,
        )
        return response
    except Exception:
        duration = perf_counter() - start
        REQUEST_COUNT.labels(method=request.method,
                             endpoint=request.url.path,
                             http_status=500).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        logger.exception("Unhandled error for path=%s", request.url.path)
        raise


class HeartFeatures(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HeartFeatures):
    data = features.dict()
    ordered_cols = numeric_features + categorical_features
    df = pd.DataFrame([[data[col] for col in ordered_cols]],
                      columns=ordered_cols)

    try:
        proba = float(model.predict_proba(df)[0, 1])
        pred = int(model.predict(df)[0])

        PREDICTION_COUNT.labels(status="success").inc()

        logger.info(
                    "predict_request age=%s sex=%s cp=%s trestbps=%s"
                    "chol=%s pred=%s proba=%.4f",
                    data["age"],
                    data["sex"],
                    data["cp"],
                    data["trestbps"],
                    data["chol"],
                    pred,
                    proba)

        return {
            "prediction": pred,
            "probability": proba,
        }
    except Exception:
        PREDICTION_COUNT.labels(status="error").inc()
        logger.exception("Error during prediction")
        raise


@app.get("/metrics")
def metrics():
    # Expose Prometheus metrics in text format
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
