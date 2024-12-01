import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'stress_detection_IT_professionals_dataset.csv'
data = pd.read_csv(file_path)

# Features and target
X = data.drop(columns=["Stress_Level"])  # Drop the target column
y = data["Stress_Level"]

# Normalize numerical columns for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a RandomForest classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_scaled, y)

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
stress_prediction = model.predict(user_input_scaled)[0]

# Display result
st.subheader("Prediction Result")
st.write(f"Predicted Stress Level: **{stress_prediction}**")