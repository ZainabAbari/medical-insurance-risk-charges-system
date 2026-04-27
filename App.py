import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load pipelines 
risk_model = joblib.load("best_model.pkl")
charges_model = joblib.load("best_model2.pkl")

st.title("Medical Insurance Risk & Charges Predictor")
st.subheader("Enter Customer Data")

age = st.number_input("Age", 18, 100, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Children", 0, 10, 0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

input_df = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex': [sex],
    'smoker': [smoker],
    'region': [region]})

if st.button("Predict"):
    risk = risk_model.predict(input_df)[0]
    charges = charges_model.predict(input_df)[0]

    st.write("Risk: ", "High " if risk == 1 else "Low")
    st.write("Estimated Charges:", round(charges, 2))