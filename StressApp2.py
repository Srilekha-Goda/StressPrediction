import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_scaled, y)
    # Save the model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

# Streamlit App Configuration
st.set_page_config(page_title="Stress Level Tracker", page_icon="🌿", layout="wide")

# Custom Styles for Ultimate Design
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap');

        body {
            background: linear-gradient(135deg, #EDF7F3, #E6F4EF);
            font-family: 'Playfair Display', serif;
        }
        .block-container {
            padding: 2rem;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.1);
        }
        .header {
            text-align: center;
            font-size: 36px;
            color: #2F4D40;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .recommendation-card {
            padding: 1.5rem;
            margin-top: 1.5rem;
            background: rgba(127, 191, 166, 0.2);
            border-radius: 15px;
            text-align: center;
            font-family: 'Lato', sans-serif;
        }
        .altair-chart {
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>🌿 Stress Level Tracker</div>", unsafe_allow_html=True)

# Sidebar for User Details
st.sidebar.header("Track Stress Over Time")
name = st.sidebar.text_input("Your Name", placeholder="Enter your name")
age = st.sidebar.number_input("Your Age", min_value=1, max_value=100, value=25)
email = st.sidebar.text_input("Your Email", placeholder="Enter your email")

if st.sidebar.button("Submit"):
    st.sidebar.success("Details Saved! Ready to track your stress? 🌿")

# Input Form
st.header("Input Your Stress Indicators")
col1, col2 = st.columns(2)

with col1:
    heart_rate = st.slider("Heart Rate (BPM)", 50, 150, 80, help="Typical resting heart rate is 60-100 BPM.")
    skin_conductivity = st.slider("Skin Conductivity (μS)", 1.0, 10.0, 5.0, step=0.1, help="Higher values may indicate stress.")

with col2:
    hours_worked = st.slider("Hours Worked Today", 0, 16, 8, help="Number of hours spent working today.")
    emails_sent = st.slider("Emails Sent Today", 0, 100, 30, help="Number of emails sent during work.")
    meetings_attended = st.slider("Meetings Attended Today", 0, 10, 3, help="Number of meetings attended today.")

# Process User Input
features = {
    "Heart_Rate": heart_rate,
    "Skin_Conductivity": skin_conductivity,
    "Hours_Worked": hours_worked,
    "Emails_Sent": emails_sent,
    "Meetings_Attended": meetings_attended,
}
user_input = pd.DataFrame([features])
user_input_scaled = scaler.transform(user_input)

# Prediction
stress_prediction = model.predict(user_input_scaled)[0]

# Map Predictions to Stress Levels
stress_levels = ["Low", "Moderate", "Above Moderate", "High", "Very High"]
stress_level = stress_levels[stress_prediction // 6]

# Display Results
st.markdown(f"<div class='recommendation-card'><h3>Your Stress Level: {stress_level}</h3><p>Personalized Tip: <i>{stress_level_map[stress_level]}</i></p></div>", unsafe_allow_html=True)

# Dynamic Visualization with Altair
chart_data = pd.DataFrame({
    "Metric": ["Heart Rate", "Skin Conductivity", "Hours Worked", "Emails Sent", "Meetings Attended"],
    "Value": [heart_rate, skin_conductivity, hours_worked, emails_sent, meetings_attended]
})
chart = alt.Chart(chart_data).mark_bar().encode(
    x="Value",
    y=alt.Y("Metric", sort="-x"),
    color=alt.condition(
        alt.datum.Value > 60, alt.value("#FF6347"), alt.value("#7FBFA6")
    )
).properties(title="Your Stress Metrics")

st.altair_chart(chart, use_container_width=True)

# Footer
st.markdown("<div class='footer'>© 2024 Stress Tracker | Stay Relaxed 🌿</div>", unsafe_allow_html=True)
