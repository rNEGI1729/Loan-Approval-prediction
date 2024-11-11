import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and preprocessor
model = load_model('loan_model.keras')  # Keras model saved with .keras extension
preprocessor = joblib.load('preprocessor.pkl')  # Preprocessor saved with joblib

# Set up the Streamlit page
st.title("Loan Prediction App")
st.write("Enter the applicant details to predict loan status")

# Input fields for user data
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=1000, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=1000, value=1000)
loan_amount = st.number_input("Loan Amount", min_value=100, value=200)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12, value=360)
credit_history = st.selectbox("Credit History", [1, 0])  # 1 for good, 0 for bad
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})

# Convert categorical variables to numerical values (One-Hot or Label Encoding)
binary_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Property_Area', 'Education']

# Using LabelEncoder for categorical columns
label_encoder = LabelEncoder()
for col in binary_columns:
    input_data[col] = label_encoder.fit_transform(input_data[col])

# Normalize the numerical features
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
scaler = StandardScaler()
input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])

# Convert the DataFrame to a NumPy array
input_data_array = input_data.values

# Prediction
prediction = model.predict(input_data_array)
loan_status = "Approved" if prediction[0] > 0.5 else "Not Approved"

# Display the prediction result
if st.button("Predict Loan Status"):
    st.write(f"The loan status is: {loan_status}")
