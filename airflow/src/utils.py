import os
import json
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import requests
import tempfile

from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from minio import Minio

# MinIO / S3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
client = MlflowClient()
BUCKET_LOGS = "pipeline-logs"

# Credenciales MinIO
os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Opcional pero recomendado para MinIO
os.environ["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


MYSQL_HOST = os.environ.get("MYSQL_HOST", "mysql-db")
MYSQL_PORT = os.environ.get("MYSQL_PORT", "3306")
MYSQL_DB = os.environ.get("MYSQL_DATABASE")
MYSQL_USER = os.environ.get("MYSQL_USER", "mlops_user")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "mlops_pass")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
#MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "diabetes_readmission_experiment")
MLFLOW_EXPERIMENT = (
    "real_estate_price_prediction"
)
MODEL_NAME = "real_estate_price_model"

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)
mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000") 
MINIO_ACCESS_KEY = os.environ.get("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD")

MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "mlflow-artifacts")

TARGET_COLUMN = "target"

DROP_COLUMNS = [
    "readmitted",
    TARGET_COLUMN
]




# ============================================================
# ⚙️ CONFIGURACIÓN GENERAL
# ============================================================

# -----------------------------
# API CONFIG
# -----------------------------
#API_BASE_URL = os.environ.get("API_BASE_URL", "http://fastapi:8003")
API_BASE_URL = "http://get-data-api:8003"


HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
BATCH_ENDPOINT = f"{API_BASE_URL}/batch"

# -----------------------------
# DATABASE CONFIG
# -----------------------------


DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

# -----------------------------
# PIPELINE CONFIG
# -----------------------------
BATCH_SIZE = 1000

RAW_TABLE = os.environ.get("RAW_TABLE")
CLEAN_TABLE = os.environ.get("CLEAN_TABLE")
SPLIT_TABLE = os.environ.get("SPLIT_TABLE")

# porcentaje splits
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# =========================
# CONFIG
# =========================

API_BASE_URL = "http://get-data-api:8003"
GROUP_NUMBER = 3

MYSQL_USER = "mlops_user"
MYSQL_PASSWORD = "mlops_pass"
MYSQL_HOST = "mysql-db"
MYSQL_PORT = "3306"
MYSQL_DATABASE = "mlops_db"

ENGINE = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)


FEATURES = [

    "status",
    "city",
    "state",
    "zip_code",

    "bed",
    "bath",
    "acre_lot",
    "house_size",
    "years_since_last_sale"
]

TARGET = "price"

CATEGORICAL_FEATURES = [

    "status",
    "city",
    "state",
    "zip_code"
]



from datetime import datetime
import pandas as pd


def save_batch_statistic(
    batch_number,
    feature_name,
    mean_value=None,
    std_value=None,
    new_categories_count=None
):

    row = pd.DataFrame([{
        "batch_number": batch_number,
        "feature_name": feature_name,
        "mean_value": mean_value,
        "std_value": std_value,
        "new_categories_count": new_categories_count,
        "used_for_training": False,
        "created_at": datetime.now()
    }])

    row.to_sql(
        "batch_statistics",
        ENGINE,
        if_exists="append",
        index=False
    )


from sqlalchemy import text


def create_tables():

    ddl_raw = """
    CREATE TABLE IF NOT EXISTS raw_data (

        raw_id BIGINT AUTO_INCREMENT PRIMARY KEY,

        batch_number INT,
        ingestion_timestamp DATETIME,

        brokered_by DOUBLE,
        status VARCHAR(100),
        price DOUBLE,
        bed DOUBLE,
        bath DOUBLE,
        acre_lot DOUBLE,
        street DOUBLE,
        city VARCHAR(255),
        state VARCHAR(255),
        zip_code VARCHAR(30),
        house_size DOUBLE,
        prev_sold_date VARCHAR(50)
    );
    """

    ddl_clean = """
    CREATE TABLE IF NOT EXISTS clean_data (

        clean_id BIGINT AUTO_INCREMENT PRIMARY KEY,

        raw_id BIGINT,
        batch_number INT,
        processed_timestamp DATETIME,

        brokered_by DOUBLE,
        status VARCHAR(100),
        price DOUBLE,
        bed DOUBLE,
        bath DOUBLE,
        acre_lot DOUBLE,
        street DOUBLE,
        city VARCHAR(255),
        state VARCHAR(255),
        zip_code VARCHAR(30),
        house_size DOUBLE,

        prev_sold_date DATE,
        years_since_last_sale DOUBLE,

        dataset_split VARCHAR(20)
    );
    """

    ddl_monitoring = """
    CREATE TABLE IF NOT EXISTS data_monitoring (

        monitoring_id BIGINT AUTO_INCREMENT PRIMARY KEY,

        batch_number INT,
        check_type VARCHAR(100),
        feature_name VARCHAR(100),
        metric_name VARCHAR(100),

        metric_value DOUBLE NULL,

        details TEXT,

        created_at DATETIME
    );
    """

    ddl_batch_statistics = """
    CREATE TABLE IF NOT EXISTS batch_statistics (

        stat_id BIGINT AUTO_INCREMENT PRIMARY KEY,

        batch_number INT NOT NULL,

        feature_name VARCHAR(100) NOT NULL,

        mean_value DOUBLE NULL,

        std_value DOUBLE NULL,

        new_categories_count INT NULL,

        used_for_training BOOLEAN DEFAULT FALSE,

        created_at DATETIME
    );
    """
    ddl_training_runs = """
    CREATE TABLE IF NOT EXISTS training_runs (
    
        run_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
        triggering_batch INT,
    
        mlflow_run_id VARCHAR(255),
    
        candidate_rmse DOUBLE,
        candidate_mae DOUBLE,
        candidate_r2 DOUBLE,
    
        promoted BOOLEAN,
    
        created_at DATETIME
    );
    """
    with ENGINE.begin() as conn:

        conn.execute(text(ddl_raw))
        conn.execute(text(ddl_clean))
        conn.execute(text(ddl_monitoring))
        conn.execute(text(ddl_batch_statistics))
        conn.execute(text(ddl_training_runs))

    print("Tables verified.")


def log_monitoring(
    batch_number,
    check_type,
    feature_name,
    metric_name,
    metric_value=None,
    details=None
):

    row = pd.DataFrame([{
        "batch_number": batch_number,
        "check_type": check_type,
        "feature_name": feature_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "details": details,
        "created_at": datetime.now()
    }])

    row.to_sql(
        "data_monitoring",
        ENGINE,
        if_exists="append",
        index=False
    )



def start():

    print("Checking API health...")

    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Health check failed: {response.status_code}"
        )

    print("API health OK")


def fetch_and_store_raw_batch():

    response = requests.get(
        f"{API_BASE_URL}/data",
        params={"group_number": GROUP_NUMBER},
        timeout=60
    )

    response.raise_for_status()

    payload = response.json()

    batch_number = payload["batch_number"]

    df = pd.DataFrame(payload["data"])

    if df.empty:
        raise ValueError("Batch returned empty")

    df["batch_number"] = batch_number
    df["ingestion_timestamp"] = datetime.now()

    if "zip" in df.columns:
        df.rename(columns={"zip": "zip_code"}, inplace=True)

    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype(str)

    df.to_sql(
        "raw_data",
        ENGINE,
        if_exists="append",
        index=False
    )

    print(
        f"Stored {len(df)} rows for batch {batch_number}"
    )

    return batch_number


def validate_schema():

    query = """
    SELECT MAX(batch_number)
    FROM raw_data
    """

    batch_number = pd.read_sql(query, ENGINE).iloc[0,0]

    df = pd.read_sql(
        f"""
        SELECT *
        FROM raw_data
        WHERE batch_number={batch_number}
        """,
        ENGINE
    )

    required_columns = [
        "brokered_by",
        "status",
        "price",
        "bed",
        "bath",
        "acre_lot",
        "street",
        "city",
        "state",
        "zip_code",
        "house_size",
        "prev_sold_date"
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        log_monitoring(
            batch_number,
            "schema_validation",
            "dataset",
            "missing_columns",
            details=json.dumps(missing)
        )

        raise ValueError(
            f"Missing columns: {missing}"
        )

    log_monitoring(
        batch_number,
        "schema_validation",
        "dataset",
        "schema_ok"
    )

    print("Schema validation OK")


def validate_data_quality():

    batch_number = pd.read_sql(
        "SELECT MAX(batch_number) batch_number FROM raw_data",
        ENGINE
    ).iloc[0,0]

    df = pd.read_sql(
        f"""
        SELECT *
        FROM raw_data
        WHERE batch_number={batch_number}
        """,
        ENGINE
    )

    duplicates = df.duplicated().sum()

    log_monitoring(
        batch_number,
        "quality_validation",
        "dataset",
        "duplicates",
        duplicates
    )

    for col in df.columns:

        nulls = int(df[col].isna().sum())

        log_monitoring(
            batch_number,
            "quality_validation",
            col,
            "null_count",
            nulls
        )

    invalid_price = (df["price"] <= 0).sum()

    log_monitoring(
        batch_number,
        "quality_validation",
        "price",
        "invalid_price",
        int(invalid_price)
    )

    print("Quality validation completed")


def get_latest_batch():

    return pd.read_sql(
        """
        SELECT MAX(batch_number) AS batch_number
        FROM raw_data
        """,
        ENGINE
    ).iloc[0, 0]



def detect_new_categories():

    categorical_cols = ["brokered_by",
                        "street",
                        "status",
                        "city",
                        "state",
                        "zip_code"
                    ]

    batch_number = pd.read_sql(
        """
        SELECT MAX(batch_number) AS batch_number
        FROM raw_data
        """,
        ENGINE
    ).iloc[0, 0]

    current = pd.read_sql(
        f"""
        SELECT *
        FROM raw_data
        WHERE batch_number = {batch_number}
        """,
        ENGINE
    )

    history = pd.read_sql(
        f"""
        SELECT *
        FROM raw_data
        WHERE batch_number < {batch_number}
        """,
        ENGINE
    )

    rows = []

    for col in categorical_cols:

        current_values = set(
            current[col]
            .dropna()
            .astype(str)
        )

        if history.empty:

            # primer batch:
            # todas las categorías son nuevas
            new_categories_count = len(current_values)

        else:

            historical_values = set(
                history[col]
                .dropna()
                .astype(str)
            )

            new_categories_count = len(
                current_values - historical_values
            )

        rows.append({
            "batch_number": batch_number,
            "feature_name": col,
            "mean_value": None,
            "std_value": None,
            "new_categories_count": new_categories_count,
            "used_for_training": False,
            "created_at": pd.Timestamp.now()
        })

    pd.DataFrame(rows).to_sql(
        "batch_statistics",
        ENGINE,
        if_exists="append",
        index=False
    )

    print(
        f"Stored category statistics for batch {batch_number}"
    )


def detect_data_drift():

    numeric_cols = ["price",
                    "bed",
                    "bath",
                    "acre_lot",
                    "house_size"
                ]

    batch_number = pd.read_sql(
        """
        SELECT MAX(batch_number) AS batch_number
        FROM raw_data
        """,
        ENGINE
    ).iloc[0, 0]

    current = pd.read_sql(
        f"""
        SELECT *
        FROM raw_data
        WHERE batch_number = {batch_number}
        """,
        ENGINE
    )

    rows = []

    for col in numeric_cols:

        rows.append({
                "batch_number": batch_number,
                "feature_name": col,
                "mean_value": current[col].mean(),
                "std_value": current[col].std(),
                "new_categories_count": None,
                "used_for_training": False,
                "created_at": pd.Timestamp.now()
            })

    pd.DataFrame(rows).to_sql(
        "batch_statistics",
        ENGINE,
        if_exists="append",
        index=False
    )

    print(
        f"Stored numerical statistics for batch {batch_number}"
    )


def preprocess_data():

    batch_number = pd.read_sql(
        "SELECT MAX(batch_number) batch_number FROM raw_data",
        ENGINE
    ).iloc[0,0]

    raw_df = pd.read_sql(
        f"""
        SELECT *
        FROM raw_data
        WHERE batch_number={batch_number}
        """,
        ENGINE
    )

    df = raw_df.copy()

    numeric_cols = [
        "brokered_by",
        "price",
        "bed",
        "bath",
        "acre_lot",
        "street",
        "house_size"
    ]

    categorical_cols = [
        "status",
        "city",
        "state",
        "zip_code"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in categorical_cols:

        mode = (
            df[col].mode().iloc[0]
            if not df[col].mode().empty
            else "UNKNOWN"
        )

        df[col] = df[col].fillna(mode)

    df["prev_sold_date"] = pd.to_datetime(
        df["prev_sold_date"],
        errors="coerce"
    )

    today = pd.Timestamp.today()

    df["years_since_last_sale"] = (
        (today - df["prev_sold_date"]).dt.days
        / 365.25
    )

    df["years_since_last_sale"] = (
        df["years_since_last_sale"]
        .fillna(
            df["years_since_last_sale"].median()
        )
    )

    before = len(df)

    df = df.drop_duplicates()

    df = df[
        (df["price"] > 0)
        & (df["house_size"] > 0)
        & (df["bed"] >= 0)
        & (df["bath"] >= 0)
        & (df["acre_lot"] >= 0)
    ]

    removed = before - len(df)

    log_monitoring(
        batch_number,
        "preprocessing",
        "dataset",
        "rows_removed",
        removed
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42
    )

    train_df["dataset_split"] = "train"
    valid_df["dataset_split"] = "validation"
    test_df["dataset_split"] = "test"

    clean_df = pd.concat(
        [
            train_df,
            valid_df,
            test_df
        ],
        ignore_index=True
    )

    clean_df["processed_timestamp"] = datetime.now()

    cols = [
        "raw_id",
        "batch_number",
        "processed_timestamp",
        "brokered_by",
        "status",
        "price",
        "bed",
        "bath",
        "acre_lot",
        "street",
        "city",
        "state",
        "zip_code",
        "house_size",
        "prev_sold_date",
        "years_since_last_sale",
        "dataset_split"
    ]

    clean_df[cols].to_sql(
        "clean_data",
        ENGINE,
        if_exists="append",
        index=False
    )

    print(
        f"Inserted {len(clean_df)} rows into clean_data"
    )



def preprocess_data():

    batch_number = pd.read_sql(
        "SELECT MAX(batch_number) batch_number FROM raw_data",
        ENGINE
    ).iloc[0,0]

    raw_df = pd.read_sql(
        f"""
        SELECT *
        FROM raw_data
        WHERE batch_number={batch_number}
        """,
        ENGINE
    )

    df = raw_df.copy()

    numeric_cols = [
        "brokered_by",
        "price",
        "bed",
        "bath",
        "acre_lot",
        "street",
        "house_size"
    ]

    categorical_cols = [
        "status",
        "city",
        "state",
        "zip_code"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in categorical_cols:

        mode = (
            df[col].mode().iloc[0]
            if not df[col].mode().empty
            else "UNKNOWN"
        )

        df[col] = df[col].fillna(mode)

    df["prev_sold_date"] = pd.to_datetime(
        df["prev_sold_date"],
        errors="coerce"
    )

    today = pd.Timestamp.today()

    df["years_since_last_sale"] = (
        (today - df["prev_sold_date"]).dt.days
        / 365.25
    )

    df["years_since_last_sale"] = (
        df["years_since_last_sale"]
        .fillna(
            df["years_since_last_sale"].median()
        )
    )

    before = len(df)

    df = df.drop_duplicates()

    df = df[
        (df["price"] > 0)
        & (df["house_size"] > 0)
        & (df["bed"] >= 0)
        & (df["bath"] >= 0)
        & (df["acre_lot"] >= 0)
    ]

    removed = before - len(df)

    log_monitoring(
        batch_number,
        "preprocessing",
        "dataset",
        "rows_removed",
        removed
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42
    )

    train_df["dataset_split"] = "train"
    valid_df["dataset_split"] = "validation"
    test_df["dataset_split"] = "test"

    clean_df = pd.concat(
        [
            train_df,
            valid_df,
            test_df
        ],
        ignore_index=True
    )

    clean_df["processed_timestamp"] = datetime.now()

    cols = [
        "raw_id",
        "batch_number",
        "processed_timestamp",
        "brokered_by",
        "status",
        "price",
        "bed",
        "bath",
        "acre_lot",
        "street",
        "city",
        "state",
        "zip_code",
        "house_size",
        "prev_sold_date",
        "years_since_last_sale",
        "dataset_split"
    ]

    clean_df[cols].to_sql(
        "clean_data",
        ENGINE,
        if_exists="append",
        index=False
    )

    print(
        f"Inserted {len(clean_df)} rows into clean_data"
    )


def simulate_dag_until_preprocessing():

    print("=" * 60)
    print("START DAG")
    print("=" * 60)

    start()

    batch_number = fetch_and_store_raw_batch()

    validate_schema()

    validate_data_quality()

    detect_new_categories()

    detect_data_drift()

    preprocess_data()

    print("=" * 60)
    print(
        f"PIPELINE COMPLETED - BATCH {batch_number}"
    )
    print("=" * 60)




# ============================== TRAIN =====================


from minio import Minio
from io import BytesIO
from datetime import datetime

PIPELINE_LOG_BUCKET = "pipeline-logs"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)


def ensure_log_bucket():

    if not minio_client.bucket_exists(
        PIPELINE_LOG_BUCKET
    ):
        minio_client.make_bucket(
            PIPELINE_LOG_BUCKET
        )



def ensure_minio_buckets():

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    for bucket in [
        MINIO_BUCKET,
        BUCKET_LOGS
    ]:

        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

    print("MinIO buckets verified")



def append_pipeline_log(text_message):

    ensure_log_bucket()

    object_name = "pipeline_history.txt"

    try:

        response = minio_client.get_object(
            PIPELINE_LOG_BUCKET,
            object_name
        )

        current_content = (
            response.read()
            .decode("utf-8")
        )

    except:

        current_content = ""

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    new_content = (
        current_content
        + "\n"
        + "=" * 80
        + "\n"
        + timestamp
        + "\n\n"
        + text_message
        + "\n"
    )

    data = new_content.encode("utf-8")

    minio_client.put_object(
        PIPELINE_LOG_BUCKET,
        object_name,
        BytesIO(data),
        len(data),
        content_type="text/plain"
    )



def decide_training():

    NUMERIC_FEATURES = [
        "price",
        "bed",
        "bath",
        "acre_lot",
        "house_size"
    ]

    CATEGORICAL_FEATURES = [
        "status",
        "city",
        "state",
        "zip_code"
    ]

    DRIFT_THRESHOLD = 0.25
    MIN_NUMERIC_DRIFTS = 2

    reasons = []

    champion = get_current_champion()

    # --------------------------------------------------
    # Primer entrenamiento
    # --------------------------------------------------

    if champion is None:

        return {
            "should_train": True,
            "decision": "TRAIN",
            "reasons": [
                "No Champion model found."
            ]
        }

    current_batch = pd.read_sql(
        """
        SELECT MAX(batch_number) AS batch_number
        FROM clean_data
        """,
        ENGINE
    ).iloc[0, 0]

    # --------------------------------------------------
    # Batches usados por el Champion
    # --------------------------------------------------

    training_stats = pd.read_sql(
        """
        SELECT *
        FROM batch_statistics
        WHERE used_for_training = TRUE
        """,
        ENGINE
    )

    if training_stats.empty:

        return {
            "should_train": True,
            "decision": "TRAIN",
            "reasons": [
                "No training reference found."
            ]
        }

    current_stats = pd.read_sql(
        f"""
        SELECT *
        FROM batch_statistics
        WHERE batch_number = {current_batch}
        """,
        ENGINE
    )

    if current_stats.empty:

        return {
            "should_train": False,
            "decision": "SKIP",
            "reasons": [
                f"No statistics found for batch {current_batch}"
            ]
        }

    for col in ["mean_value", "std_value"]:

        training_stats[col] = pd.to_numeric(
            training_stats[col],
            errors="coerce"
        )

        current_stats[col] = pd.to_numeric(
            current_stats[col],
            errors="coerce"
        )

    reasons.append(
        f"Current batch={current_batch}"
    )

    reference_batches = sorted(
        training_stats["batch_number"]
        .unique()
        .tolist()
    )

    reasons.append(
        f"Reference batches={reference_batches}"
    )

    # --------------------------------------------------
    # NUMERIC DRIFT
    # --------------------------------------------------

    numeric_drifts = 0

    for feature in NUMERIC_FEATURES:

        ref_rows = training_stats[
            training_stats["feature_name"] == feature
        ]

        curr_rows = current_stats[
            current_stats["feature_name"] == feature
        ]

        if ref_rows.empty or curr_rows.empty:
            continue

        ref_mean = ref_rows[
            "mean_value"
        ].mean()

        ref_std = ref_rows[
            "std_value"
        ].mean()

        curr_mean = curr_rows.iloc[0][
            "mean_value"
        ]

        curr_std = curr_rows.iloc[0][
            "std_value"
        ]

        # Mean drift

        if (
            pd.notna(ref_mean)
            and pd.notna(curr_mean)
            and abs(ref_mean) > 0
        ):

            mean_drift = abs(
                curr_mean - ref_mean
            ) / abs(ref_mean)

            if mean_drift > DRIFT_THRESHOLD:

                numeric_drifts += 1

                reasons.append(
                    f"{feature}: mean drift "
                    f"{mean_drift:.2%}"
                )

        # Std drift

        if (
            pd.notna(ref_std)
            and pd.notna(curr_std)
            and ref_std > 0
        ):

            std_drift = abs(
                curr_std - ref_std
            ) / ref_std

            if std_drift > DRIFT_THRESHOLD:

                numeric_drifts += 1

                reasons.append(
                    f"{feature}: std drift "
                    f"{std_drift:.2%}"
                )

    train_numeric = (
        numeric_drifts >= MIN_NUMERIC_DRIFTS
    )

    # --------------------------------------------------
    # CATEGORICAL DRIFT
    # --------------------------------------------------

    train_categorical = False

    current_data = pd.read_sql(
        f"""
        SELECT
            status,
            city,
            state,
            zip_code
        FROM clean_data
        WHERE batch_number = {current_batch}
        """,
        ENGINE
    )

    historical_data = pd.read_sql(
        """
        SELECT
            c.status,
            c.city,
            c.state,
            c.zip_code
        FROM clean_data c
        INNER JOIN
        (
            SELECT DISTINCT batch_number
            FROM batch_statistics
            WHERE used_for_training = TRUE
        ) t
        ON c.batch_number = t.batch_number
        """,
        ENGINE
    )

    for feature in CATEGORICAL_FEATURES:

        historical_categories = set(
            historical_data[feature]
            .dropna()
            .astype(str)
        )

        current_categories = set(
            current_data[feature]
            .dropna()
            .astype(str)
        )

        unseen = (
            current_categories
            - historical_categories
        )

        if len(unseen) > 0:

            train_categorical = True

            reasons.append(
                f"{feature}: "
                f"{len(unseen)} unseen categories"
            )

    # --------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------

    should_train = (
        train_numeric
        or train_categorical
    )

    if not should_train:

        reasons.append(
            "No numeric drift above threshold."
        )

        reasons.append(
            "No unseen categories detected."
        )

    return {
        "should_train": should_train,
        "decision": (
            "TRAIN"
            if should_train
            else "SKIP"
        ),
        "reasons": reasons
    }



def skip_training(decision):

    text = [
        "",
        "TRAINING DECISION",
        "",
        "Result:",
        "SKIP TRAINING",
        "",
        "Reasons:",
        ""
    ]

    for reason in decision["reasons"]:
        text.append(
            f"- {reason}"
        )

    append_pipeline_log(
        "\n".join(text)
    )


def load_training_dataset():

    query = """
    SELECT *
    FROM clean_data
    """

    df = pd.read_sql(
        query,
        ENGINE
    )

    return df


def build_training_metadata(df):

    metadata = {

        "numeric_ranges": {},

        "allowed_categories": {}
    }

    numeric_features = [

        "bed",
        "bath",
        "acre_lot",
        "house_size",
        "years_since_last_sale"
    ]

    for col in numeric_features:

        metadata["numeric_ranges"][col] = {

            "min": float(
                df[col].min()
            ),

            "max": float(
                df[col].max()
            )
        }

    for col in CATEGORICAL_FEATURES:

        metadata[
            "allowed_categories"
        ][col] = sorted(
            df[col]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )

    return metadata


def get_training_reason():

    decision = decide_training()

    return decision.get(
        "reasons",
        ["No reason available"]
    )

import subprocess


def get_git_commit():

    try:

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            )
            .decode()
            .strip()
        )

    except Exception:

        return "unknown"

def train_evaluate_register_candidate():

    NUMERIC_FEATURES = [
        "price",
        "bed",
        "bath",
        "acre_lot",
        "house_size"
    ]
 
    CATEGORICAL_FEATURES = [
        "status",
        "city",
        "state",
        "zip_code"
    ]
    df = load_training_dataset()

    train_df = df[
        df["dataset_split"] == "train"
    ].copy()

    test_df = df[
        df["dataset_split"] == "test"
    ].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # ==================================================
    # TRAINING CONTEXT
    # ==================================================

    batch_numbers = sorted(
        df["batch_number"]
        .unique()
        .tolist()
    )

    training_reason = (
        get_training_reason()
    )

    # ==================================================
    # GIT COMMIT
    # ==================================================

    try:

        repo_path = os.getcwd()

        subprocess.run(
            [
                "git",
                "config",
                "--global",
                "--add",
                "safe.directory",
                repo_path
            ],
            check=False,
            capture_output=True
        )

        git_commit = (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD"
                ],
                cwd=repo_path
            )
            .decode()
            .strip()
        )

    except Exception:

        git_commit = "unknown"

    # ==================================================
    # MODEL
    # ==================================================

    model = CatBoostRegressor(

        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        verbose=False,
        random_seed=42
    )

    with mlflow.start_run() as run:

        # ==============================================
        # TRAIN
        # ==============================================

        model.fit(
            X_train,
            y_train,
            cat_features=CATEGORICAL_FEATURES
        )

        predictions = model.predict(
            X_test
        )

        # ==============================================
        # METRICS
        # ==============================================

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mae",
            mae
        )

        mlflow.log_metric(
            "r2",
            r2
        )

        # ==============================================
        # PARAMS
        # ==============================================

        mlflow.log_params({

            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "RMSE",

            "categorical_features":
                ",".join(
                    CATEGORICAL_FEATURES
                ),

            "numeric_features":
                ",".join(
                    FEATURES
                )
        })

        # ==============================================
        # TAGS
        # ==============================================

        mlflow.set_tags({

            "git_commit":
                git_commit,

            "training_reason":
                " | ".join(
                    training_reason
                ),

            "batch_numbers":
                ",".join(
                    map(
                        str,
                        batch_numbers
                    )
                )
        })

        # ==============================================
        # EXISTING METADATA
        # ==============================================

        metadata = build_training_metadata(
            df
        )

        # ==============================================
        # TRAINING CONTEXT
        # ==============================================

        training_context = {

            "git_commit":
                git_commit,

            "batch_numbers":
                batch_numbers,

            "training_reason":
                training_reason,

            "training_metrics": {

                "rmse":
                    float(rmse),

                "mae":
                    float(mae),

                "r2":
                    float(r2)
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:

            # ------------------------------------------
            # metadata existente
            # ------------------------------------------

            metadata_path = os.path.join(
                tmpdir,
                "training_metadata.json"
            )

            with open(
                metadata_path,
                "w"
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4
                )

            mlflow.log_artifact(
                metadata_path
            )

            # ------------------------------------------
            # nuevo contexto
            # ------------------------------------------

            context_path = os.path.join(
                tmpdir,
                "training_context.json"
            )

            with open(
                context_path,
                "w"
            ) as f:

                json.dump(
                    training_context,
                    f,
                    indent=4
                )

            mlflow.log_artifact(
                context_path
            )

        # ==============================================
        # MODEL
        # ==============================================

        model_info = mlflow.catboost.log_model(
            model,
            artifact_path="model"
        )

        run_id = run.info.run_id

    return {

        "run_id":
            run_id,

        "model_uri":
            model_info.model_uri,

        "rmse":
            rmse,

        "mae":
            mae,

        "r2":
            r2
    }



def register_candidate_model(
    candidate
):

    registered_model = (
        mlflow.register_model(
            model_uri=candidate[
                "model_uri"
            ],
            name=MODEL_NAME
        )
    )

    return registered_model.version



def get_current_champion():

    try:

        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            "champion"
        )

        run_id = champion.run_id

        run = client.get_run(run_id)

        rmse = run.data.metrics.get("rmse")

        return {
            "version": champion.version,
            "run_id": run_id,
            "rmse": rmse
        }

    except Exception:

        return None
    



def get_champion_rmse(
    champion_version
):

    version = (
        client.get_model_version(
            MODEL_NAME,
            champion_version
        )
    )

    run_id = version.run_id

    run = mlflow.get_run(
        run_id
    )

    return run.data.metrics[
        "rmse"
    ]



def compare_and_decide_promotion(
    candidate_metrics
):

    champion = get_current_champion()

    # Primer modelo

    if champion is None:

        return {
            "promote": True,
            "reason": (
                "No Champion model found. "
                "First model automatically promoted."
            ),
            "champion_rmse": None,
            "candidate_rmse": candidate_metrics["rmse"],
            "improvement_pct": None
        }

    champion_rmse = champion["rmse"]
    candidate_rmse = candidate_metrics["rmse"]

    # Protección por si champion no tiene rmse

    if champion_rmse is None:

        return {
            "promote": True,
            "reason": (
                "Champion model exists but "
                "RMSE metric was not found. "
                "Candidate promoted."
            ),
            "champion_rmse": None,
            "candidate_rmse": candidate_rmse,
            "improvement_pct": None
        }

    improvement = (
        (champion_rmse - candidate_rmse)
        / champion_rmse
    )

    promote = (
        candidate_rmse < champion_rmse
    )

    if promote:

        reason = (
            f"Promotion approved. "
            f"Candidate RMSE={candidate_rmse:.4f} "
            f"improves Champion RMSE={champion_rmse:.4f}. "
            f"Relative improvement={improvement:.2%}."
        )

    else:

        reason = (
            f"Promotion rejected. "
            f"Candidate RMSE={candidate_rmse:.4f} "
            f"is not better than Champion RMSE={champion_rmse:.4f}. "
            f"Relative improvement={improvement:.2%}."
        )

    return {
        "promote": promote,
        "reason": reason,
        "champion_rmse": champion_rmse,
        "candidate_rmse": candidate_rmse,
        "improvement_pct": improvement
    }


def promote_model(
    version,
    candidate_metrics,
    promotion
):

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        version
    )

    current_batch = pd.read_sql(
        """
        SELECT MAX(batch_number)
        AS batch_number
        FROM clean_data
        """,
        ENGINE
    ).iloc[0, 0]

    with ENGINE.begin() as conn:

        conn.execute(
            text(
                """
                UPDATE batch_statistics
                SET used_for_training = TRUE
                WHERE used_for_training = FALSE
                """
            )
        )

    training_run = pd.DataFrame([{

        "triggering_batch":
            current_batch,

        "mlflow_run_id":
            candidate_metrics["run_id"],

        "candidate_rmse":
            candidate_metrics["rmse"],

        "candidate_mae":
            candidate_metrics["mae"],

        "candidate_r2":
            candidate_metrics["r2"],

        "promoted":
            True,

        "created_at":
            pd.Timestamp.now()
    }])

    training_run.to_sql(

        "training_runs",

        ENGINE,

        if_exists="append",

        index=False
    )

    improvement = promotion["improvement_pct"]
    if improvement is None:
        improvement_text = "N/A"
    else:
        improvement_text = f"{improvement:.2%}"

    append_pipeline_log(
f"""
PROMOTION DECISION

Decision:
PROMOTE

Reason:
{promotion['reason']}

Champion RMSE:
{promotion['champion_rmse']}

Candidate RMSE:
{promotion['candidate_rmse']}

Improvement:
{improvement_text}

Version:
{version}
"""
    )

    print(
        f"Champion updated -> {version}"
    )


def reject_model(
    candidate_metrics,
    promotion
):

    current_batch = pd.read_sql(
        """
        SELECT MAX(batch_number)
        AS batch_number
        FROM clean_data
        """,
        ENGINE
    ).iloc[0, 0]

    training_run = pd.DataFrame([{

        "triggering_batch":
            current_batch,

        "mlflow_run_id":
            candidate_metrics["run_id"],

        "candidate_rmse":
            candidate_metrics["rmse"],

        "candidate_mae":
            candidate_metrics["mae"],

        "candidate_r2":
            candidate_metrics["r2"],

        "promoted":
            False,

        "created_at":
            pd.Timestamp.now()
    }])

    training_run.to_sql(

        "training_runs",

        ENGINE,

        if_exists="append",

        index=False
    )

    append_pipeline_log(
f"""
PROMOTION DECISION

Decision:
REJECT

Reason:
{promotion['reason']}

Champion RMSE:
{promotion['champion_rmse']}

Candidate RMSE:
{promotion['candidate_rmse']}

Improvement:
{promotion['improvement_pct']:.2%}
"""
)

    print(
        "Candidate rejected"
    )


def end_pipeline():

    append_pipeline_log(
        "Pipeline finished successfully"
    )

    print(
        "=" * 80
    )

    print(
        "PIPELINE FINISHED"
    )

    print(
        "=" * 80
    )


def simulate_training_dag():
    print("Tracking URI:")
    print(mlflow.get_tracking_uri())

    print(
        "\nSTART TRAINING DAG\n"
    )

    decision = decide_training()

    if not decision["should_train"]:
    
        skip_training(
            decision
        )
    
        end_pipeline()
    
        return
    
    append_pipeline_log(
        "\n".join(
            [
                "TRAINING DECISION",
                "",
                "Result:",
                "TRAIN",
                "",
                "Reasons:",
                *[
                    f"- {r}"
                    for r in decision["reasons"]
                ]
            ]
        )
    )

    ensure_minio_buckets()
    
    candidate_metrics = (
        train_evaluate_register_candidate()
    )
    
    version = register_candidate_model(
        candidate_metrics
    )
    
    promotion = (
        compare_and_decide_promotion(
            candidate_metrics
        )
    )
    
    if promotion["promote"]:
    
        promote_model(
            version,
            candidate_metrics,
            promotion
        )
    
    else:
    
        reject_model(
            candidate_metrics,
            promotion
        )
    
    end_pipeline()
