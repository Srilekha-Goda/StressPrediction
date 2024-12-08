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
LOGO_FILE = "Logo.png"  # Ensure Logo.png is in the same directory

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
st.set_page_config(page_title="Stress Level Tracker", page_icon=LOGO_FILE, layout="centered")

# Custom Styles for Modern UI
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap');

        * {
            font-family: 'Playfair Display', serif;
        }
        body {
            background: linear-gradient(135deg, #EDF7F3, #E6F4EF);
        }
        .block-container {
            padding: 2rem;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.1);
        }
        .stSidebar {
            background-color: #EFF8F5;
            padding: 2rem 1rem;
            border-right: 2px solid #BFDCD1;
        }
        .stButton button {
            background: linear-gradient(135deg, #7FBFA6, #6EA98B);
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 10px 15px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #6EA98B, #5E8C76);
            transform: scale(1.05);
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.2);
        }
        .header {
            text-align: center;
            font-size: 36px;
            color: #2F4D40;
            font-weight: bold;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .footer {
            font-size: 0.9rem;
            text-align: center;
            color: #6E9A87;
            font-family: 'Lato', sans-serif;
        }
        .card {
            background: rgba(127, 191, 166, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.1);
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #E0EFE8;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .low { background: #7FBFA6; }
        .moderate { background: #FFD700; }
        .high { background: #FF6347; }
        .very-high { background: #FF4500; }
    </style>
""", unsafe_allow_html=True)

# Display Logo and Header Using st.image
col1, col2 = st.columns([1, 6])
with col1:
    st.image(LOGO_FILE, width=80)
with col2:
    st.markdown("<h1 style='text-align: left; font-family: Playfair Display; color: #2F4D40;'>Stress Level Tracker</h1>", unsafe_allow_html=True)

# Sidebar for User Details
st.sidebar.image(LOGO_FILE, use_column_width=True)
st.sidebar.header("Welcome")
st.sidebar.markdown("Track and manage your stress levels effortlessly!")
name = st.sidebar.text_input("Enter your name")
age = st.sidebar.number_input("Your age", min_value=1, max_value=100, value=25)
email = st.sidebar.text_input("Email Address")

if st.sidebar.button("Submit"):
    st.sidebar.success("Your details have been saved. Welcome!")

# Input Form for Stress Data
st.header("Input Your Stress Indicators")
col1, col2 = st.columns(2)

with col1:
    heart_rate = st.slider("Heart Rate (BPM)", 50, 150, 80, help="Typical resting heart rate is between 60-100 BPM.")
    skin_conductivity = st.slider("Skin Conductivity (μS)", 1.0, 10.0, 5.0, step=0.1, help="Higher values may indicate stress.")

with col2:
    hours_worked = st.slider("Hours Worked Today", 0, 16, 8, help="Number of hours spent working today.")
    emails_sent = st.slider("Emails Sent Today", 0, 100, 30, help="Number of emails sent during work.")
    meetings_attended = st.slider("Meetings Attended Today", 0, 10, 3, help="Number of meetings attended today.")

# User Input Processing
features = {
    "Heart_Rate": heart_rate,
    "Skin_Conductivity": skin_conductivity,
    "Hours_Worked": hours_worked,
    "Emails_Sent": emails_sent,
    "Meetings_Attended": meetings_attended,
}
user_input = pd.DataFrame([features])
user_input_scaled = scaler.transform(user_input)

# Stress Prediction
stress_prediction = model.predict(user_input_scaled)[0]

# Map Stress Prediction to Levels and Recommendations
stress_level_map = {
    "Low": "Keep up the great work! You're in a calm and balanced state.",
    "Moderate": "You're doing okay, but consider taking short breaks.",
    "Above Moderate": "Stress is building up. Practice mindfulness or breathing exercises.",
    "High": "Your stress is elevated. Try relaxing activities like meditation.",
    "Very High": "Seek professional guidance for stress management if necessary."
}

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

# Display Results with Modern Card and Progress Bar
stress_color = {
    "Low": "low",
    "Moderate": "moderate",
    "Above Moderate": "moderate",
    "High": "high",
    "Very High": "very-high"
}

st.markdown(f"""
<div class="card">
    <h3>Your Stress Level: <b style="color: {stress_color[stress_level]}">{stress_level}</b></h3>
    <div class="progress-bar">
        <div class="progress-bar-fill {stress_color[stress_level]}" style="width: {stress_prediction * 10}%"></div>
    </div>
    <p><i>{stress_level_map[stress_level]}</i></p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
    <div class="footer">
        <img src="{LOGO_FILE}" width="50" style="vertical-align:middle;"> © 2024 Stress Tracker App | Designed to help you manage stress effectively
    </div>
""", unsafe_allow_html=True)





