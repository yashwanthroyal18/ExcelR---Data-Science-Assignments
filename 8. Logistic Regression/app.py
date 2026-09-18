"""
Streamlit app for deploying the Logistic Regression diabetes prediction model.
Run locally with:  streamlit run app.py
"""
import joblib
import numpy as np
import streamlit as st

# Load the trained model and scaler saved from the notebook
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="\U0001FA7A")
st.title("\U0001FA7A Diabetes Risk Predictor")
st.write(
    "This app uses a Logistic Regression model trained on the Pima Indians "
    "Diabetes dataset to estimate the probability that a patient has diabetes, "
    "based on standard diagnostic measurements."
)

st.header("Enter Patient Measurements")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
with col2:
    insulin = st.number_input("Insulin (mu U/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"High risk: the model predicts **Diabetes** "
                  f"(probability = {probability:.1%}).")
    else:
        st.success(f"Low risk: the model predicts **No Diabetes** "
                   f"(probability of diabetes = {probability:.1%}).")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "This tool is for educational purposes only and is not a substitute "
        "for professional medical advice or diagnosis."
    )
