# ============================================================
# IMPORTS
# ============================================================

import os
import random
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://api-inference:8001"
)

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Real Estate Price Predictor",
    layout="wide"
)

st.title(
    "🏠 Real Estate Price Predictor (cambio)"
)

# ============================================================
# HEALTH
# ============================================================

try:

    health = requests.get(
        f"{API_URL}/health",
        timeout=5
    ).json()

except Exception:

    st.error(
        "❌ Could not connect to API"
    )

    st.stop()

if health["status"] != "ok":

    st.warning(
        "⚠️ No model loaded"
    )

    st.stop()

# ============================================================
# METADATA
# ============================================================

metadata = requests.get(
    f"{API_URL}/metadata",
    timeout=10
).json()

feature_metadata = metadata["metadata"]

allowed_categories = (
    feature_metadata[
        "allowed_categories"
    ]
)

numeric_ranges = (
    feature_metadata[
        "numeric_ranges"
    ]
)

# ============================================================
# MODEL INFO
# ============================================================

st.success(
    f"""
    Model: real_estate_price_model

    Version: {metadata["version"]}
    """
)

# ============================================================
# RANDOM PAYLOAD
# ============================================================

if (
    "example_payload"
    not in st.session_state
):

    st.session_state.example_payload = {}

def load_random_payload():

    try:

        sample = requests.get(
            f"{API_URL}/sample",
            timeout=10
        ).json()

        st.session_state.example_payload = (
            sample["payload"]
        )

    except Exception as e:

        st.error(str(e))

st.button(
    "🎲 Generate Example",
    on_click=load_random_payload
)

# ============================================================
# FORM
# ============================================================

payload = {}

with st.form("predict_form"):

    st.subheader(
        "Property Features"
    )

    cols = st.columns(3)

    all_features = (
        list(
            allowed_categories.keys()
        )
        +
        list(
            numeric_ranges.keys()
        )
    )

    for idx, feature in enumerate(
        all_features
    ):

        default = (
            st.session_state
            .example_payload
            .get(feature)
        )

        with cols[idx % 3]:

            # =====================================
            # CATEGORICAL
            # =====================================

            if feature in allowed_categories:

                values = (
                    allowed_categories[
                        feature
                    ]
                )

                selected_index = 0

                if default in values:

                    selected_index = (
                        values.index(default)
                    )

                payload[feature] = (
                    st.selectbox(
                        feature,
                        values,
                        index=selected_index
                    )
                )

            # =====================================
            # NUMERIC
            # =====================================

            else:

                limits = numeric_ranges[
                    feature
                ]

                min_value = float(
                    limits["min"]
                )

                max_value = float(
                    limits["max"]
                )

                if default is None:

                    default = (
                        min_value
                        + max_value
                    ) / 2

                payload[feature] = (
                    st.number_input(
                        feature,
                        min_value=min_value,
                        max_value=max_value,
                        value=float(default)
                    )
                )

    submitted = st.form_submit_button(
        "🔮 Predict Price"
    )

# ============================================================
# PREDICTION
# ============================================================

if submitted:

    with st.spinner(
        "Running inference..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

            if response.status_code != 200:

                st.error(
                    response.text
                )

            else:

                result = response.json()

                st.success(
                    "✅ Prediction completed"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Estimated Price",
                        f"${result['predicted_price']:,.2f}"
                    )

                with col2:

                    st.metric(
                        "Model Version",
                        result[
                            "model_version"
                        ]
                    )

                    st.metric(
                        "Latency (ms)",
                        result[
                            "processing_time_ms"
                        ]
                    )

                st.subheader(
                    "Request"
                )

                st.json(payload)

                st.subheader(
                    "Response"
                )

                st.json(result)

        except Exception as e:

            st.error(str(e))