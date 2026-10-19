
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ChurnSense | Customer Analytics",
    page_icon="📊",
    layout="wide",
)

ARTIFACT_PATH = Path(r"D:\Webots\churnsense_artifact.joblib")


# --------------------------------------------------
# 2. Load the trained model and metadata
# --------------------------------------------------

@st.cache_resource
def load_artifact(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Run the artifact-saving cell in your notebook first."
        )

    return joblib.load(path)


st.title("📊 ChurnSense")
st.subheader("Telecom Customer Churn Prediction")

st.write(
    "Estimate churn probability for an individual customer "
    "to help prioritize retention outreach."
)

try:
    artifact = load_artifact(ARTIFACT_PATH)
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


model = artifact["model"]
threshold = float(artifact["threshold"])
feature_columns = artifact["feature_columns"]
numeric_columns = artifact["numeric_columns"]
categorical_columns = artifact["categorical_columns"]
category_options = artifact["category_options"]
numeric_defaults = artifact["numeric_defaults"]

st.info(
    f"Classification threshold: {threshold:.3f} "
    "| Predictions are estimates, not guarantees."
)


# --------------------------------------------------
# 3. Customer input form
# --------------------------------------------------

st.markdown("### Enter customer details")

customer = {}

with st.form("customer_prediction_form"):

    numeric_features = [
        col for col in feature_columns
        if col in numeric_columns
    ]

    categorical_features = [
        col for col in feature_columns
        if col in categorical_columns
    ]

    st.markdown("#### Numeric features")

    numeric_grid = st.columns(3)

    for i, feature in enumerate(numeric_features):
        with numeric_grid[i % 3]:
            default_value = float(
                numeric_defaults.get(feature, 0.0)
            )

            customer[feature] = st.number_input(
                label=feature,
                min_value=0.0,
                value=max(0.0, default_value),
                step=1.0,
                key=f"numeric_{feature}",
            )

    st.markdown("#### Categorical features")

    categorical_grid = st.columns(3)

    for i, feature in enumerate(categorical_features):
        options = category_options.get(feature, [])

        with categorical_grid[i % 3]:
            if options:
                customer[feature] = st.selectbox(
                    label=feature,
                    options=options,
                    key=f"categorical_{feature}",
                )
            else:
                st.error(
                    f"No training categories available for {feature}."
                )

    submitted = st.form_submit_button(
        "🔍 Predict churn risk",
        use_container_width=True,
    )


# --------------------------------------------------
# 4. Validate input and predict
# --------------------------------------------------

if submitted:

    # Preserve the exact feature order used during training.
    customer_df = pd.DataFrame(
        [customer],
        columns=feature_columns,
    )

    # Convert numeric features explicitly.
    for feature in numeric_columns:
        customer_df[feature] = pd.to_numeric(
            customer_df[feature],
            errors="coerce",
        )

    if customer_df.isnull().any().any():
        st.error(
            "Some inputs are missing or invalid. "
            "Please check all customer details."
        )
        st.stop()

    try:
        probabilities = model.predict_proba(customer_df)[0]

        # Find the probability belonging to the churn class.
        classes = list(model.classes_)

        if 1 in classes:
            churn_index = classes.index(1)
        elif "Yes" in classes:
            churn_index = classes.index("Yes")
        else:
            st.error(
                f"Could not identify the churn class. "
                f"Model classes: {classes}"
            )
            st.stop()

        churn_probability = float(probabilities[churn_index])
        predicted_churn = churn_probability >= threshold

        st.divider()
        st.markdown("### Prediction results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated churn probability",
            f"{churn_probability:.1%}",
        )

        col2.metric(
            "Classification threshold",
            f"{threshold:.1%}",
        )

        col3.metric(
            "Prediction",
            "Higher churn risk" if predicted_churn
            else "Lower churn risk",
        )

        st.progress(
            min(max(churn_probability, 0.0), 1.0)
        )

        if predicted_churn:
            st.warning(
                "This customer is above the model's classification "
                "threshold. Consider reviewing their needs and "
                "possible retention options."
            )
        else:
            st.success(
                "This customer is below the model's classification "
                "threshold."
            )

        with st.expander("Review submitted customer data"):
            st.dataframe(
                customer_df,
                use_container_width=True,
            )

        st.caption(
            "This prediction supports decision-making; it does not "
            "guarantee that a customer will or will not churn."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that the saved model artifact and dashboard "
            "use compatible versions of pandas and scikit-learn."
        )
