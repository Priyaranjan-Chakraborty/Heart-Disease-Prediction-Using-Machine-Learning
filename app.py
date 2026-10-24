import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Load the saved model, scaler, and training feature columns
# Keep these three .pkl files in the same folder as app.py.
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Logistic Regression.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's information below. The trained Logistic Regression "
    "model will estimate whether the input is classified as heart disease or no heart disease."
)
st.caption(
    "Educational project only. This tool is not a medical diagnosis and must not "
    "be used to make healthcare decisions."
)

try:
    model, scaler, columns = load_artifacts()
except Exception as exc:
    st.error(
        "Could not load the model files. Make sure these files are in the same "
        "folder as app.py: `Logistic Regression.pkl`, `scaler.pkl`, and `columns.pkl`."
    )
    st.exception(exc)
    st.stop()

# Inputs use the feature categories present in the notebook.
with st.form("heart_disease_form"):
    st.subheader("Patient information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        sex = st.selectbox("Sex", ["Female", "Male"])
        chest_pain = st.selectbox(
            "Chest pain type",
            [
                "asymptomatic",
                "atypical angina",
                "non-anginal",
                "typical angina",
            ],
        )
        resting_bp = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50.0,
            max_value=250.0,
            value=130.0,
            step=1.0,
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=50.0,
            max_value=700.0,
            value=220.0,
            step=1.0,
        )
        fasting_blood_sugar = st.selectbox(
            "Fasting blood sugar greater than 120 mg/dl?",
            ["No", "Yes"],
        )

    with col2:
        resting_ecg = st.selectbox(
            "Resting ECG result",
            ["lv hypertrophy", "normal", "st-t abnormality"],
        )
        max_heart_rate = st.number_input(
            "Maximum heart rate achieved",
            min_value=40.0,
            max_value=250.0,
            value=150.0,
            step=1.0,
        )
        exercise_angina = st.selectbox("Exercise-induced angina?", ["No", "Yes"])
        st_depression = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
        major_vessels = st.selectbox(
            "Number of major blood vessels",
            [0, 1, 2, 3],
        )
        st_slope = st.selectbox(
            "ST segment slope",
            ["downsloping", "flat", "upsloping"],
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    # Start with the numeric columns that were retained during training.
    input_data = {
        "Age": int(age),
        "Resting_Blood_Pressure": int(resting_bp),
        "Cholesterol_Level": int(cholesterol),
        "Fasting_Blood_Sugar": int(fasting_blood_sugar == "Yes"),
        "Maxaimum_Heart_Rate": int(max_heart_rate),
        "Exercise_Induced_Angina": int(exercise_angina == "Yes"),
        "ST_Depression": int(st_depression),
        "Major_Blood_Vessels": int(major_vessels),
        # Drop-first encoding: the first/alphabetical category is represented
        # by all-zero dummy values.
        "Sex_Male": int(sex == "Male"),
        "Chest_Pain_Type_atypical angina": int(chest_pain == "atypical angina"),
        "Chest_Pain_Type_non-anginal": int(chest_pain == "non-anginal"),
        "Chest_Pain_Type_typical angina": int(chest_pain == "typical angina"),
        "Resting_ECG_Result_normal": int(resting_ecg == "normal"),
        "Resting_ECG_Result_st-t abnormality": int(resting_ecg == "st-t abnormality"),
        "ST_Segment_Slope_flat": int(st_slope == "flat"),
        "ST_Segment_Slope_upsloping": int(st_slope == "upsloping"),
    }

    # Match the exact feature order saved during model training.
    input_df = pd.DataFrame([input_data]).reindex(columns=columns, fill_value=0)
    input_df = input_df.astype(int)

    try:
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])

        st.subheader("Prediction result")
        if prediction == 1:
            st.error("The model predicts: Heart disease indicated (class 1).")
        else:
            st.success("The model predicts: No heart disease indicated (class 0).")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            class_labels = list(model.classes_)
            probability_map = dict(zip(class_labels, probabilities))
            if 0 in probability_map and 1 in probability_map:
                st.write(f"Model probability for class 0: **{probability_map[0] * 100:.1f}%**")
                st.write(f"Model probability for class 1: **{probability_map[1] * 100:.1f}%**")
                st.caption(
                    "These are model probabilities for this dataset, not a clinically "
                    "validated estimate of an individual's medical risk."
                )

        with st.expander("View encoded input sent to the model"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error("Prediction failed. Check that the saved model and preprocessing files match.")
        st.exception(exc)
