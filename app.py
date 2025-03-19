# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:51:26 2025

@author: User
"""

import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Diabetes Risk Prediction (Krishantini")

st.write("Enter the required features to get a prediction:")

# Binary Inputs (Yes = 1, No = 0)
high_blood_pressure = st.radio("Do you have High Blood Pressure?", ["No", "Yes"])
high_cholesterol = st.radio("Do you have High Cholesterol?", ["No", "Yes"])
cholesterol_check_5yrs = st.radio("Did you check Cholesterol in the last 5 years?", ["No", "Yes"])
smoker = st.radio("Are you a smoker?", ["No", "Yes"])
had_stroke = st.radio("Have you had a Stroke before?", ["No", "Yes"])
heart_disease_attack = st.radio("Do you have Heart Disease or had a Heart Attack?", ["No", "Yes"])
had_physical_activity = st.radio("Do you engage in Physical Activity?", ["No", "Yes"])
heavy_alcohol_drinker = st.radio("Are you a Heavy Alcohol Drinker?", ["No", "Yes"])
difficulty_walking = st.radio("Do you have Difficulty Walking?", ["No", "Yes"])

# Continuous Numeric Inputs
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
general_health = st.slider("General Health (1=Poor to 5=Excellent)", 1, 5, 3)
mental_health_days = st.number_input("Mental Health (Poor days in last 30 days)", min_value=0, max_value=30, value=5)
physical_health_days = st.number_input("Physical Health (Poor days in last 30 days)", min_value=0, max_value=30, value=5)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education = st.slider("Education Level (1=No Education to 5=Graduate)", 1, 5, 3)
income = st.slider("Income Level (1=Lowest to 8=Highest)", 1, 8, 4)

# Convert binary responses to 0 and 1
binary_mapping = {"No": 0, "Yes": 1}
input_data = np.array([[
    binary_mapping[high_blood_pressure], binary_mapping[high_cholesterol], binary_mapping[cholesterol_check_5yrs], 
    bmi, binary_mapping[smoker], binary_mapping[had_stroke], binary_mapping[heart_disease_attack], 
    binary_mapping[had_physical_activity], binary_mapping[heavy_alcohol_drinker], 
    general_health, mental_health_days, physical_health_days, binary_mapping[difficulty_walking], 
    age, education, income
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Prediction: {'Hypertension' if prediction[0] == 1 else 'No Hypertension'}")