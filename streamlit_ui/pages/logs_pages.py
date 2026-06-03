from minio import Minio
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")

MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")

MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")

PIPELINE_BUCKET = "pipeline-logs"

client = Minio(
    f"http://{MINIO_ENDPOINT}",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def read_pipeline_log():

    try:

        response = client.get_object(
            PIPELINE_BUCKET,
            "pipeline_log.txt"
        )

        content = (
            response.read()
            .decode("utf-8")
        )

        response.close()
        response.release_conn()

        return content

    except Exception as e:

        return f"Error reading log: {e}"

def get_latest_log():

    objects = list(
        client.list_objects(
            "pipeline-logs",
            recursive=True
        )
    )

    if not objects:
        return "No logs found"

    latest = sorted(
        objects,
        key=lambda x: x.last_modified,
        reverse=True
    )[0]

    response = client.get_object(
        "pipeline-logs",
        latest.object_name
    )

    content = (
        response.read()
        .decode("utf-8")
    )

    response.close()
    response.release_conn()

    return (
        latest.object_name,
        content
    )

import streamlit as st

st.set_page_config(
    page_title="Pipeline Logs",
    layout="wide"
)

st.title(
    "📜 Training Pipeline Logs"
)

if st.button(
    "🔄 Refresh"
):

    st.rerun()

filename, log_content = get_latest_log()

st.subheader(filename)

st.code(
    log_content,
    language="text"
)