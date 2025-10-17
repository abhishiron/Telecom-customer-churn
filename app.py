import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler # Added for clarity, though technically only needed if re-fitting

# --- 1. CONFIGURATION AND ASSET LOADING ---

# Use st.cache_resource to load large, static assets only once.
@st.cache_resource
def load_assets():
    try:
        # Load your assets. 'encoder' is a dictionary of LabelEncoders.
        model = pickle.load(open('best_model.pkl', 'rb'))
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, encoder, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model asset file not found. Please ensure 'best_model.pkl', 'encoder.pkl', and 'scaler.pkl' are in the same directory as app.py. Missing file: {e}")
        return None, None, None

model, encoder, scaler = load_assets()

# --- 2. STREAMLIT UI LAYOUT (No changes needed here) ---

st.set_page_config(layout="wide", page_title="Customer Churn Prediction")
st.title("Customer Churn Prediction 📊")

if model is None:
    st.stop() # Stop execution if assets failed to load

# Create the three-column layout to mimic the HTML structure
col1, col2, col3 = st.columns(3)

# Dictionary to hold all user inputs
user_input = {}

# --- COLUMN 1: Personal and Basic Service Info ---
with col1:
    st.header("Customer Information")
    user_input['gender'] = st.selectbox("Gender", ["Male", "Female"], index=0)
    # Note: The model expects 'SeniorCitizen' as 0 or 1. We handle the UI mapping here.
    senior_citizen_map = {"No": 0, "Yes": 1}
    senior_citizen_ui = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
    user_input['SeniorCitizen'] = senior_citizen_map[senior_citizen_ui]
    
    user_input['Partner'] = st.selectbox("Partner", ["Yes", "No"], index=0)
    user_input['Dependents'] = st.selectbox("Dependents", ["Yes", "No"], index=0)
    
    # Numerical input for Tenure
    user_input['tenure'] = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12, step=1)
    user_input['PhoneService'] = st.selectbox("Phone Service", ["Yes", "No"], index=0)

# --- COLUMN 2: Internet/Security Services ---
with col2:
    st.header("Service Details")
    user_input['MultipleLines'] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=1)
    user_input['InternetService'] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=0)
    
    # Options are identical for these internet-related features
    internet_options = ["Yes", "No", "No internet service"]
    user_input['OnlineSecurity'] = st.selectbox("Online Security", internet_options, index=1)
    user_input['OnlineBackup'] = st.selectbox("Online Backup", internet_options, index=1)
    user_input['DeviceProtection'] = st.selectbox("Device Protection", internet_options, index=1)
    user_input['TechSupport'] = st.selectbox("Tech Support", internet_options, index=1)

# --- COLUMN 3: Streaming, Billing, and Charges ---
with col3:
    st.header("Billing & Payments")
    user_input['StreamingTV'] = st.selectbox("Streaming TV", internet_options, index=1)
    user_input['StreamingMovies'] = st.selectbox("Streaming Movies", internet_options, index=1)
    
    user_input['Contract'] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
    user_input['PaperlessBilling'] = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
    
    user_input['PaymentMethod'] = st.selectbox("Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 
        index=0
    )

    # Numerical inputs for charges
    user_input['MonthlyCharges'] = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)
    user_input['TotalCharges'] = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=0.01)

# --- 3. PREDICTION LOGIC (Corrected) ---

# The final list of features in the order your model expects (from notebook output)
FINAL_COLUMNS_ORDER = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

def preprocess_and_predict(input_data, model, encoders, scaler):
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])
    
    # 1. Apply Label Encoding to categorical columns
    # The 'encoders' object is a dictionary of fitted LabelEncoder objects.
    for col, le in encoders.items():
        # LabelEncoder expects a 1D array/Series, so we apply it directly.
        # It's critical to only transform the columns that were originally fitted.
        if col in df.columns:
            df[col] = le.transform(df[col]) 
    
    # 2. Apply Scaling to numerical features
    # StandardScaler expects numerical columns.
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
    
    # 3. Final Features Preparation
    # Select and order the features exactly as expected by the model
    final_features = df[FINAL_COLUMNS_ORDER].values
    
    # 4. Make Prediction
    prediction = model.predict(final_features)
    # Get the probability for the positive class (churn = 1)
    probability = model.predict_proba(final_features)[:, 1]
    
    return prediction[0], probability[0]


# Prediction button placed at the end of the form
st.markdown("---")
if st.button("Predict Churn", type="primary"):
    
    # Edge case: Total Charges cannot be less than Monthly Charges (unless tenure is 0/1)
    if user_input['TotalCharges'] < user_input['MonthlyCharges'] and user_input['tenure'] > 1:
        st.warning("⚠️ Warning: Total Charges are usually greater than Monthly Charges times Tenure. Please check the input.")
    
    try:
        # Pass the loaded 'encoder' dictionary to the function
        churn_class, churn_proba = preprocess_and_predict(user_input, model, encoder, scaler)
        
        st.subheader("Prediction Result")

        if churn_class == 1:
            st.error(f"❌ **The customer is likely to CHURN!**")
            st.write(f"Confidence (Probability of Churn): **{churn_proba:.2f}**")
        else:
            st.success(f"✅ **The customer is NOT likely to churn.**")
            st.write(f"Confidence (Probability of No Churn): **{1 - churn_proba:.2f}**")
            
        st.balloons()
        
    except Exception as e:
        # Now st.exception will provide a useful traceback
        st.exception(f"An error occurred during prediction. Check the data preprocessing steps: {e}")