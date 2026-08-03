import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/multiple_linear_regression_model.pkl")

st.title("🏥 Medical Insurance Cost Prediction")

st.write("Enter your details below.")

# Inputs
age = st.number_input("Age", 18, 100, 30)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    10.0,
    60.0,
    25.0
)

children = st.number_input(
    "Children",
    0,
    10,
    0
)

smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)

region = st.selectbox(
    "Region",
    [
        "Northeast",
        "Northwest",
        "Southeast",
        "Southwest"
    ]
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "cat__sex_male": [1 if sex == "Male" else 0],
        "cat__smoker_yes": [1 if smoker == "Yes" else 0],
        "cat__region_northwest": [1 if region == "Northwest" else 0],
        "cat__region_southeast": [1 if region == "Southeast" else 0],
        "cat__region_southwest": [1 if region == "Southwest" else 0],
        "remainder__age": [age],
        "remainder__bmi": [bmi],
        "remainder__children": [children]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Insurance Cost: ${prediction[0]:,.2f}"
    )
