import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Diabetes Profile",
    page_icon="🩺",
)

st.title("🩺 Diabetes Profile")
st.write("Enter patient measurements to identify their profile.")

st.info(
    "Educational prototype: results are based on patient clusters, "
    "not a diabetes diagnosis."
)

st.caption(
    "Leave unavailable measurements blank. Age and Diabetes Pedigree "
    "Function are required. Missing measurements are estimated by the model."
)


# Patient form
with st.form("patient_form"):
    left, right = st.columns(2)

    with left:
        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            value=None,
            step=1.0,
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            value=None,
            step=1.0,
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            value=None,
            step=1.0,
        )

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            value=None,
            step=1.0,
        )

    with right:
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=None,
            step=0.1,
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=None,
            step=0.01,
            format="%.3f",
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            value=None,
            step=1,
        )

    submitted = st.form_submit_button("Predict")


# Send the form to FastAPI
if submitted:
    if age is None or pedigree is None:
        st.error("Please enter Age and Diabetes Pedigree Function.")
        st.stop()

    measurements = [
        glucose, blood_pressure, skin_thickness, insulin, bmi
    ]

    if all(value is None or value == 0 for value in measurements):
        st.error("Please provide at least one measured clinical value.")
        st.stop()

    patient = {
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }

    try:
        with st.spinner("Predicting..."):
            response = requests.post(
                f"{API_URL.rstrip('/')}/predict",
                json=patient,
                timeout=30,
            )

        if response.status_code == 422:
            st.error("Some inputs are invalid. Check your measurements.")
            st.stop()

        response.raise_for_status()
        result = response.json()

        risk_category = result["risk_category"].strip().lower()
        cluster = result["cluster"]

    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")
        st.stop()

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Make sure it is running.")
        st.stop()

    except requests.exceptions.RequestException:
        st.error("The API returned an error. Check the FastAPI terminal.")
        st.stop()

    except (ValueError, KeyError, TypeError):
        st.error("The API returned an unexpected response.")
        st.stop()

    st.subheader("Prediction result")
    st.write(f"Patient cluster: **{cluster}**")

    if risk_category == "high risk":
        st.error("🔴 High risk — cluster-based category")

    elif risk_category == "low risk":
        st.success("🟢 Low risk — cluster-based category")

    else:
        st.warning(f"Unrecognized risk category: {risk_category}")

    st.write(
        "Follow-up: have a qualified clinician review the measurements "
        "and decide whether further assessment is needed. "
        "A low-risk cluster does not rule out diabetes."
    )

    if any(value is None or value == 0 for value in measurements):
        st.caption(
            "Some measurements were missing or zero. "
            "The pipeline replaced them with training medians."
        )




# python -m streamlit run frontend/app.py