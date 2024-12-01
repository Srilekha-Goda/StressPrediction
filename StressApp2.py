import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
# File paths
MODEL_FILE = "stress_model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = 'stress_detection_IT_professionals_dataset.csv'
# Load dataset
data = pd.read_csv(DATA_FILE)
# Features and target
X = data.drop(columns=["Stress_Level"])
y = data["Stress_Level"]
# Check if model and scaler are already saved
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
else:
    # Normalize numerical columns for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train a RandomForest classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)       #This is the main machine learning model that we have used
    model.fit(X_scaled, y)
    # Save the model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
# Streamlit App
st.sidebar.header("User Information")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
email = st.sidebar.text_input("Email")
st.sidebar.button("Submit")
st.title("Stress Level Prediction")
st.markdown("""
Provide input values below to predict your stress level based on physiological and work-related parameters.
""")
# Input fields for prediction
features = {}
features["Heart_Rate"] = st.slider("Heart Rate", 50, 150, 80)
features["Skin_Conductivity"] = st.slider("Skin Conductivity", 1.0, 10.0, 5.0, step=0.1)
features["Hours_Worked"] = st.slider("Hours Worked", 0, 20, 8)
features["Emails_Sent"] = st.slider("Emails Sent", 0, 100, 30)
features["Meetings_Attended"] = st.slider("Meetings Attended", 0, 20, 5)
# Preprocess input
user_input = pd.DataFrame([features])
user_input_scaled = scaler.transform(user_input)
# Prediction
print(model.predict(user_input_scaled)[0])
print(model.predict(user_input_scaled))
stress_prediction = model.predict(user_input_scaled)[0]
# Map prediction to stress levels
if stress_prediction <= 20:
    stress_level = "Low"
elif 21 <= stress_prediction <= 23:
    stress_level = "Moderate"
elif 24 <= stress_prediction <= 26:
    stress_level = "Above Moderate"
elif 27 <= stress_prediction <= 28:
    stress_level = "High"
else:
    stress_level = "Very High"
# Display result
st.subheader("Prediction Result")
st.write(f"Predicted Stress Level: **{stress_prediction}**")
st.write(f"Stress Category: **{stress_level}**")