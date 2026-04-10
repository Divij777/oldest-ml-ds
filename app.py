import streamlit as st
import joblib
import pandas as pd

# 1. Load the model and feature list
try:
    data = joblib.load("insurance_model.joblib")
    # Check if data is a dict or just the model
    if isinstance(data, dict):
        model = data['model']
        features = data['features']
    else:
        model = data
        features = [] # Fallback to current kernel features if available
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None
    features = []

st.set_page_config(page_title="Insurance Cost Predictor", layout="wide")
st.title("Medical Cost Predictor")

# 2. User Inputs (Organized into columns for a better UI)
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Patient Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    city_type = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])
    
with col2:
    st.header("Health & Lifestyle")
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=40000, value=8000)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    
    st.subheader("Conditions")
    diabetes = st.checkbox("Diabetes")
    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    asthma = st.checkbox("Asthma")

with col3:
    st.header("Medical History & Insurance")
    doctor_visits_per_year = st.number_input("Doctor Visits (per year)", min_value=0, max_value=50, value=2)
    hospital_admissions = st.number_input("Hospital Admissions", min_value=0, max_value=20, value=0)
    medication_count = st.number_input("Medication Count", min_value=0, max_value=20, value=1)
    
    st.subheader("Insurance")
    insurance_type = st.selectbox("Insurance Type", ["None", "Private", "Government"])
    insurance_coverage_pct = st.slider("Insurance Coverage (%)", min_value=0, max_value=100, value=50)
    previous_year_cost = st.number_input("Previous Year Cost (₹)", min_value=0.0, value=5000.0)

# 3. Prediction logic
if st.button("Predict", type="primary") and model is not None:
    
    # Map checkbox booleans to 1/0 for the model
    input_dict = {
        'age': [age],
        'gender': [gender],
        'bmi': [bmi],
        'smoker': [smoker], # Leaving as Yes/No string as per the CSV format
        'diabetes': [1 if diabetes else 0],
        'hypertension': [1 if hypertension else 0],
        'heart_disease': [1 if heart_disease else 0],
        'asthma': [1 if asthma else 0],
        'physical_activity_level': [physical_activity_level],
        'daily_steps': [daily_steps],
        'sleep_hours': [sleep_hours],
        'stress_level': [stress_level],
        'doctor_visits_per_year': [doctor_visits_per_year],
        'hospital_admissions': [hospital_admissions],
        'medication_count': [medication_count],
        'insurance_type': [insurance_type],
        'insurance_coverage_pct': [insurance_coverage_pct],
        'city_type': [city_type],
        'previous_year_cost': [previous_year_cost]
    }
    
    input_df = pd.DataFrame(input_dict)

    # Add missing columns with 0s to match model expectations
    # (Helpful if the model requires one-hot encoded columns and you handle dummy conversion here)
    if len(features) > 0:
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        # Ensure columns are in the exact same order as training
        input_df = input_df[features]

    try:
        prediction = model.predict(input_df)
        st.success(f"Estimated 12-Month Medical Cost: **₹{prediction[0]:,.2f}**")
    except ValueError as e:
        st.error(f"Prediction Error: {e}\n\n*Tip: If your model expects numeric dummy variables for categorical fields (like gender, city_type), ensure your pipeline processes them or apply pd.get_dummies() before predicting.*")
