from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable
from datetime import datetime

from src.utils import (
    start,
    fetch_and_store_raw_batch,
    validate_schema,
    validate_data_quality,
    detect_new_categories,
    detect_data_drift,
    preprocess_data,
    decide_training,
    skip_training,
    ensure_minio_buckets,
    train_evaluate_register_candidate,
    register_candidate_model,
    compare_and_decide_promotion,
    promote_model,
    reject_model,
    end_pipeline,
    create_tables
)

default_args = {
    "owner": "airflow",
    "retries": 1,
}

# ============================================================
# FUNCIONES WRAPPER PARA LAS BIFURCACIONES
# ============================================================

def decide_training_branch(**context):
    """
    Decide si entrenar o no basado en drift detection.
    Retorna el task_id siguiente: 'skip_training' o 'ensure_minio_buckets'
    """
    decision = decide_training()
    context["task_instance"].xcom_push(key="training_decision", value=decision)
    
    if not decision["should_train"]:
        return "skip_training"
    else:
        return "ensure_minio_buckets"


def compare_and_promote_branch(**context):
    """
    Compara modelo candidato con champion.
    Retorna el task_id siguiente: 'promote_model' o 'reject_model'
    """
    # Obtener métricas del candidato del contexto
    ti = context["task_instance"]
    candidate_metrics = ti.xcom_pull(
        task_ids="train_evaluate_register_candidate",
        key="candidate_metrics"
    )
    
    promotion = compare_and_decide_promotion(candidate_metrics)
    ti.xcom_push(key="promotion_decision", value=promotion)
    
    if promotion["promote"]:
        return "promote_model"
    else:
        return "reject_model"


def register_and_push_version(**context):
    """
    Registra modelo candidato y retorna versión.
    """
    ti = context["task_instance"]
    candidate_metrics = ti.xcom_pull(
        task_ids="train_evaluate_register_candidate",
        key="candidate_metrics"
    )
    
    version = register_candidate_model(candidate_metrics)
    ti.xcom_push(key="model_version", value=version)
    return version


def promote_model_wrapper(**context):
    """
    Wrapper para promover modelo con context.
    """
    ti = context["task_instance"]
    version = ti.xcom_pull(
        task_ids="register_candidate_model",
        key="return_value"
    )
    candidate_metrics = ti.xcom_pull(
        task_ids="train_evaluate_register_candidate",
        key="candidate_metrics"
    )
    promotion = ti.xcom_pull(
        task_ids="compare_and_promote",
        key="promotion_decision"
    )
    
    promote_model(version, candidate_metrics, promotion)


def reject_model_wrapper(**context):
    """
    Wrapper para rechazar modelo con context.
    """
    ti = context["task_instance"]
    candidate_metrics = ti.xcom_pull(
        task_ids="train_evaluate_register_candidate",
        key="candidate_metrics"
    )
    promotion = ti.xcom_pull(
        task_ids="compare_and_promote",
        key="promotion_decision"
    )
    
    reject_model(candidate_metrics, promotion)


def skip_training_wrapper(**context):
    """
    Wrapper para skip training con context.
    """
    ti = context["task_instance"]
    decision = ti.xcom_pull(
        task_ids="decide_training",
        key="training_decision"
    )
    skip_training(decision)


def train_and_push_metrics(**context):
    """
    Entrena modelo y retorna métricas para usar en siguientes tareas.
    """
    candidate_metrics = train_evaluate_register_candidate()
    context["task_instance"].xcom_push(
        key="candidate_metrics",
        value=candidate_metrics
    )
    return candidate_metrics


# ============================================================
# DEFINICIÓN DEL DAG
# ============================================================

with DAG(
    dag_id="house_price_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "house_price"],
) as dag:

    # ============================================================
    # ETAPA 1: INGESTA Y PREPROCESSING
    # ============================================================
    
    task_start = PythonOperator(
        task_id="check_api_health",
        python_callable=start,
    )

    task_create_tables = PythonOperator(
        task_id="create_db_tables",
        python_callable=create_tables,
    )

    task_ingest = PythonOperator(
        task_id="fetch_and_store_raw_batch",
        python_callable=fetch_and_store_raw_batch,
    )

    task_validate_schema = PythonOperator(
        task_id="validate_schema",
        python_callable=validate_schema,
    )

    task_validate_quality = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate_data_quality,
    )

    task_detect_categories = PythonOperator(
        task_id="detect_new_categories",
        python_callable=detect_new_categories,
    )

    task_detect_drift = PythonOperator(
        task_id="detect_data_drift",
        python_callable=detect_data_drift,
    )

    task_preprocessing = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    # ============================================================
    # ETAPA 2: DECISIÓN DE ENTRENAMIENTO
    # ============================================================
    
    task_decide_training = BranchPythonOperator(
        task_id="decide_training",
        python_callable=decide_training_branch,
        provide_context=True,
    )

    # ============================================================
    # ETAPA 3: PATH A - SKIP TRAINING
    # ============================================================
    
    task_skip_training = PythonOperator(
        task_id="skip_training",
        python_callable=skip_training_wrapper,
        provide_context=True,
    )

    # ============================================================
    # ETAPA 4: PATH B - ENTRENAR MODELO
    # ============================================================
    
    task_ensure_minio = PythonOperator(
        task_id="ensure_minio_buckets",
        python_callable=ensure_minio_buckets,
    )

    task_train = PythonOperator(
        task_id="train_evaluate_register_candidate",
        python_callable=train_and_push_metrics,
        provide_context=True,
    )

    task_register = PythonOperator(
        task_id="register_candidate_model",
        python_callable=register_and_push_version,
        provide_context=True,
    )

    task_compare = BranchPythonOperator(
        task_id="compare_and_promote",
        python_callable=compare_and_promote_branch,
        provide_context=True,
    )

    # ============================================================
    # ETAPA 5: PATH B.1 - PROMOVER MODELO
    # ============================================================
    
    task_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
        provide_context=True,
    )

    # ============================================================
    # ETAPA 6: PATH B.2 - RECHAZAR MODELO
    # ============================================================
    
    task_reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model_wrapper,
        provide_context=True,
    )

    # ============================================================
    # ETAPA 7: FINALIZACIÓN (Converge todos los paths)
    # ============================================================
    
    task_end = PythonOperator(
        task_id="end_pipeline",
        python_callable=end_pipeline,
        trigger_rule="none_failed",
    )

    # ============================================================
    # DEFINIR FLUJO DEL DAG
    # ============================================================
    
    # Flujo de ingesta y preprocesamiento
    (
        task_start
        >> task_create_tables
        >> task_ingest
        >> task_validate_schema
        >> task_validate_quality
        >> task_detect_categories
        >> task_detect_drift
        >> task_preprocessing
        >> task_decide_training
    )

    # Path A: Skip training
    task_decide_training >> task_skip_training >> task_end

    # Path B: Entrenar modelo
    task_decide_training >> task_ensure_minio >> task_train >> task_register >> task_compare

    # Path B.1: Promover
    task_compare >> task_promote >> task_end

    # Path B.2: Rechazar
    task_compare >> task_reject >> task_end