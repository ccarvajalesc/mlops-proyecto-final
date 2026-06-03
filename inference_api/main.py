# ============================================================
# 📦 IMPORTS
# ============================================================

import os
import json
import time
import random
import tempfile
from datetime import datetime

import pandas as pd
import mlflow
import mlflow.catboost

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from sqlalchemy import create_engine, text

from mlflow.tracking import MlflowClient

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from prometheus_fastapi_instrumentator import Instrumentator


# ============================================================
# ⚙️ CONFIG
# ============================================================

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DATABASE", "mlflow_db")
MYSQL_USER = os.getenv("MYSQL_USER", "mlops_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mlops_pass")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MODEL_NAME = os.getenv("MODEL_NAME", "real_estate_price_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

MODEL_REFRESH_INTERVAL = 1  # segundos

print(
    "MLFLOW_TRACKING_URI=",
    MLFLOW_TRACKING_URI
)


# ============================================================
# 🗄️ DB ENGINE
# ============================================================

def create_engine_db():

    return create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )

ENGINE = create_engine_db()


# ============================================================
# 📊 PROMETHEUS
# ============================================================

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests"
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total successful predictions"
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total prediction errors"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

MODEL_VERSION_GAUGE = Gauge(
    "model_version",
    "Current model version",
    ["model", "alias"]
)


MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether a model is loaded"
)

MODEL_INFO = Gauge(
    "model_info",
    "Current loaded model",
    ["model_name", "model_version", "model_alias"]
)   

# ============================================================
# 🚀 APP
# ============================================================

app = FastAPI(title="MLflow Inference API", version="1.0")

Instrumentator().instrument(app).expose(app)


# ============================================================
# 🌎 GLOBAL STATE
# ============================================================

MODEL = None
CB_MODEL = None
FEATURE_METADATA = None

MODEL_VERSION = None
RUN_ID = None

LAST_REFRESH = 0

MODEL_FEATURES = []
CAT_FEATURE_INDICES = []

client = MlflowClient()


# ============================================================
# 🔁 LOAD MODEL
# ============================================================

def load_model():

    global MODEL
    global CB_MODEL
    global FEATURE_METADATA

    global MODEL_VERSION
    global RUN_ID

    global MODEL_FEATURES
    global CAT_FEATURE_INDICES

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    champion = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        )
    )

    MODEL_VERSION = champion.version
    RUN_ID = champion.run_id

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    MODEL = mlflow.pyfunc.load_model(
        model_uri
    )

    CB_MODEL = mlflow.catboost.load_model(
        model_uri
    )

    MODEL_FEATURES = (
        CB_MODEL.feature_names_
    )

    CAT_FEATURE_INDICES = (
        CB_MODEL.get_cat_feature_indices()
    )

    metadata_path = (
        client.download_artifacts(
            RUN_ID,
            "training_metadata.json"
        )
    )

    with open(metadata_path, "r") as f:

        FEATURE_METADATA = json.load(f)

    MODEL_VERSION_GAUGE.labels(
        model=MODEL_NAME,
        alias=MODEL_ALIAS
    ).set(float(MODEL_VERSION))

    MODEL_LOADED.set(1)

    MODEL_INFO.labels(
        model_name=MODEL_NAME,
        model_version=str(MODEL_VERSION),
        model_alias=MODEL_ALIAS
    ).set(1)

    print(
        f"Loaded model v{MODEL_VERSION}"
    )


# ============================================================
# 🔄 REFRESH MODEL (TTL)
# ============================================================

def refresh_model():

    global LAST_REFRESH

    now = time.time()

    if (
        MODEL is not None
        and
        now - LAST_REFRESH
        < MODEL_REFRESH_INTERVAL
    ):
        return

    try:

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        champion = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                MODEL_ALIAS
            )
        )

        if MODEL is None:

            load_model()

            LAST_REFRESH = now

            return

        if str(champion.version) != str(MODEL_VERSION):

            print(
                f"New model detected "
                f"{MODEL_VERSION}"
                f" -> "
                f"{champion.version}"
            )

            load_model()

        LAST_REFRESH = now

    except Exception as e:

        print(
            f"Refresh failed: {e}"
        )


# ============================================================
# ✅ VALIDATION
# ============================================================

def validate_payload(payload):

    validated = {}

    numeric_ranges = (
        FEATURE_METADATA["numeric_ranges"]
    )

    allowed_categories = (
        FEATURE_METADATA["allowed_categories"]
    )

    # ==================================================
    # CATEGORICAL FEATURES
    # ==================================================

    for feature, categories in (
        allowed_categories.items()
    ):

        if feature not in payload:

            raise HTTPException(
                status_code=400,
                detail=f"Missing feature: {feature}"
            )

        value = str(
            payload[feature]
        )

        if value not in categories:

            raise HTTPException(
                status_code=400,
                detail={
                    "feature": feature,
                    "invalid_value": value,
                    "allowed_values": categories[:20]
                }
            )

        validated[feature] = value

    # ==================================================
    # NUMERIC FEATURES
    # ==================================================

    for feature, limits in (
        numeric_ranges.items()
    ):

        if feature not in payload:

            raise HTTPException(
                status_code=400,
                detail=f"Missing feature: {feature}"
            )

        try:

            value = float(
                payload[feature]
            )

        except Exception:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid numeric value "
                    f"for {feature}"
                )
            )

        if value < limits["min"]:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{feature} below minimum "
                    f"({limits['min']})"
                )
            )

        if value > limits["max"]:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{feature} above maximum "
                    f"({limits['max']})"
                )
            )

        validated[feature] = value

    return validated


# ============================================================
# 🧠 SAMPLE PAYLOAD
# ============================================================
@app.get("/sample")
def sample():

    refresh_model()

    if FEATURE_METADATA is None:

        raise HTTPException(
            status_code=503,
            detail="Metadata not loaded"
        )

    sample = {}

    # categóricas

    for feature, categories in (
        FEATURE_METADATA["allowed_categories"]
        .items()
    ):

        sample[feature] = random.choice(
            categories
        )

    # numéricas

    for feature, limits in (
        FEATURE_METADATA["numeric_ranges"]
        .items()
    ):

        sample[feature] = round(
            random.uniform(
                limits["min"],
                limits["max"]
            ),
            4
        )

    return {
        "model_version": MODEL_VERSION,
        "payload": sample
    }

# ============================================================
# 📊 FEATURE METADATA
# ============================================================

@app.get("/metadata")
def metadata():

    refresh_model()

    if FEATURE_METADATA is None:

        raise HTTPException(
            status_code=503,
            detail="Metadata not loaded"
        )

    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "metadata": FEATURE_METADATA
    }


@app.on_event("startup")
def startup():

    print("Starting API...")

    create_inference_table()

    try:

        load_model()

    except Exception as e:

        print(
            f"Could not load model: {e}"
        )

    print("API ready")

# ============================================================
# 🔮 PREDICT
# ============================================================

@app.post("/predict")
def predict(payload: dict):

    start_time = time.time()

    REQUEST_COUNT.inc()

    try:

        refresh_model()

        if MODEL is None:

            PREDICTION_ERRORS.inc()

            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )

        validated_payload = (
            validate_payload(
                payload
            )
        )

        input_df = pd.DataFrame(
            [validated_payload]
        )

        input_df = input_df[
            MODEL_FEATURES
        ]

        for idx in CAT_FEATURE_INDICES:

            col = MODEL_FEATURES[idx]

            input_df[col] = (
                input_df[col]
                .astype(str)
            )

        prediction = float(
            MODEL.predict(
                input_df
            )[0]
        )

        processing_time_ms = (
            time.time()
            - start_time
        ) * 1000

        REQUEST_LATENCY.observe(
            processing_time_ms / 1000
        )

        PREDICTION_COUNT.inc()

        response = {

            "predicted_price": round(
                prediction,
                2
            ),

            "model_name":
                MODEL_NAME,

            "model_version":
                MODEL_VERSION,

            "model_alias":
                MODEL_ALIAS,

            "processing_time_ms":
                round(
                    processing_time_ms,
                    2
                )
        }

        log_inference(
            request_json=payload,
            response_json=response,
            predicted_price=prediction,
            processing_time_ms=processing_time_ms
        )

        return response

    except HTTPException:

        PREDICTION_ERRORS.inc()

        raise

    except Exception as e:

        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 🗄️ LOGS
# ============================================================
def create_inference_table():

    query = """
    CREATE TABLE IF NOT EXISTS inference_logs (

        id BIGINT AUTO_INCREMENT PRIMARY KEY,

        timestamp DATETIME NOT NULL,

        model_name VARCHAR(255),

        model_version VARCHAR(50),

        model_alias VARCHAR(50),

        processing_time_ms FLOAT,

        predicted_price DOUBLE,

        request_json JSON,

        response_json JSON
    )
    """

    with ENGINE.begin() as conn:

        conn.execute(text(query))

    print("Inference table ready")

create_inference_table()

def log_inference(
    request_json,
    response_json,
    predicted_price,
    processing_time_ms
):
    """
    Stores inference request into MySQL.
    """

    query = text(
        """
        INSERT INTO inference_logs (

            timestamp,
            model_name,
            model_version,
            model_alias,
            processing_time_ms,
            predicted_price,
            request_json,
            response_json

        ) VALUES (

            :timestamp,
            :model_name,
            :model_version,
            :model_alias,
            :processing_time_ms,
            :predicted_price,
            :request_json,
            :response_json
        )
        """
    )

    with ENGINE.begin() as conn:

        conn.execute(
            query,
            {
                "timestamp":
                    datetime.utcnow(),

                "model_name":
                    MODEL_NAME,

                "model_version":
                    str(MODEL_VERSION),

                "model_alias":
                    MODEL_ALIAS,

                "processing_time_ms":
                    float(processing_time_ms),

                "predicted_price":
                    float(predicted_price),

                "request_json":
                    json.dumps(request_json),

                "response_json":
                    json.dumps(response_json)
            }
        )

# ============================================================
# ❤️ HEALTH
# ============================================================

@app.get("/health")
def health():

    refresh_model()

    return {
        "status": "ok" if MODEL else "down",
        "model_version": MODEL_VERSION
    }


# ============================================================
# 📊 METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )