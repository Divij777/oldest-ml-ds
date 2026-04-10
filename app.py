import streamlit as st
import joblib
import pandas as pd

# 1. Load the model and feature list
try:
    data = joblib.load("insurance_model.joblib")
    if isinstance(data, dict):
        model = data['model']
        features = data['features']
    else:
        model = data
        features = []
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

st.set_page_config(page_title="Insurance Predictor", layout="centered")
st.title("🏥 Insurance Cost Predictor")
st.write("Enter the details below to estimate annual medical charges.")

# 2. User Inputs (Matching all features in insurance(1).csv)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Gender", ["female", "male"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# 3. Prediction logic
if st.button("Predict Cost", type="primary"):
    if model is not None:
        # Create input dataframe matching original dataset structure
        input_dict = {
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        }
        input_df = pd.DataFrame(input_dict)

        # Handle feature alignment
        # If your model was trained with one-hot encoding, this section 
        # ensures the input matches the expected 'features' list.
        if features:
            # Apply same encoding used during training (e.g., get_dummies)
            input_df_encoded = pd.get_dummies(input_df)
            
            # Add missing columns with 0s and align order
            for col in features:
                if col not in input_df_encoded.columns:
                    input_df_encoded[col] = 0
            input_df_final = input_df_encoded[features]
        else:
            input_df_final = input_df

        try:
            prediction = model.predict(input_df_final)
            # Display result in Rupees as requested
            st.success(f"### Estimated Annual Cost: ₹{prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model not loaded. Please check 'insurance_model.joblib'.")
