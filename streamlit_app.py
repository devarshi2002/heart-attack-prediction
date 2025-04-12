import streamlit as st
import numpy as np
import joblib

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

rf_r2 = 0.8301 * 100
lr_r2 = 0.3254 * 100
best_model_name = "Random Forest Regressor" if rf_r2 > lr_r2 else "Linear Regression"


st.title("Heart Attack Risk Prediction App")

st.sidebar.header("Model Performance")
st.sidebar.write(f"Random Forest R² Score: **{rf_r2:.2f}%**")
st.sidebar.write(f"Linear Regression R² Score: **{lr_r2:.2f}%**")
st.sidebar.write(f"Using Best Model: **{best_model_name}**")

st.header("Enter Patient Details")
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

sex = 1 if sex == "Male" else 0 
user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

user_data_scaled = scaler.transform(user_data)

if st.button("Predict"):
    risk_score = model.predict(user_data_scaled)[0] 
    
    risk_percent = max(0, min(risk_score, 1)) * 100

    st.write(f"**Predicted Heart Attack Risk: {risk_percent:.2f}%**")

    if risk_percent > 70:
        st.error("**High Risk of Heart Attack! Please consult a doctor immediately.**")
    elif risk_percent > 30:
        st.warning("**Moderate Risk. Consider lifestyle changes & regular check-ups.**")
    else:
        st.success("**Low Risk. Keep maintaining a healthy lifestyle!**")
