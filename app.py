import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. CONFIGURATION AND ASSET LOADING ---

# Use st.cache_resource to load large, static assets only once.
@st.cache_resource
def load_assets():
    try:
        # Adjust file paths if they are in a different directory (e.g., 'model_assets/best_model.pkl')
        model = pickle.load(open('best_model.pkl', 'rb'))
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, encoder, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model asset file not found. Please ensure 'best_model.pkl', 'encoder.pkl', and 'scaler.pkl' are in the same directory as app.py. Missing file: {e}")
        return None, None, None

model, encoder, scaler = load_assets()

# --- 2. STREAMLIT UI LAYOUT ---

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
    # Note: The model expects 'SeniorCitizen' as 0 or 1, but the UI shows 'No'/'Yes'
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
    # You might want to set realistic min/max values based on your training data
    user_input['MonthlyCharges'] = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)
    user_input['TotalCharges'] = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=0.01)

# --- 3. PREDICTION LOGIC ---

# The full list of features in the order your model expects (must match training feature names!)
# Adjust this list if your model was trained with a different order or set of columns.
FEATURE_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'gender_Male', 'gender_Female', 
    'Partner_Yes', 'Partner_No', 'Dependents_Yes', 'Dependents_No', 'PhoneService_Yes', 'PhoneService_No', 
    'MultipleLines_Yes', 'MultipleLines_No', 'MultipleLines_No phone service', 'InternetService_DSL', 
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes', 'OnlineSecurity_No', 
    'OnlineSecurity_No internet service', 'OnlineBackup_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service', 
    'DeviceProtection_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service', 'TechSupport_Yes', 
    'TechSupport_No', 'TechSupport_No internet service', 'StreamingTV_Yes', 'StreamingTV_No', 
    'StreamingTV_No internet service', 'StreamingMovies_Yes', 'StreamingMovies_No', 
    'StreamingMovies_No internet service', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 
    'PaperlessBilling_Yes', 'PaperlessBilling_No', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)'
]


def preprocess_and_predict(input_data, model, encoder, scaler, feature_columns):
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])
    
    # 1. Separate Numerical and Categorical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = df.drop(columns=numerical_cols + ['SeniorCitizen']).columns.tolist()
    
    # 2. Scale Numerical features
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # 3. Encode Categorical features
    # Use the loaded encoder to transform the categorical columns
    encoded_features = encoder.transform(df[categorical_cols])
    
    # Get feature names from the encoder (assuming it's a OneHotEncoder)
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    
    # 4. Combine all features
    # Use the pre-defined feature order (FEATURE_COLUMNS)
    final_df = pd.concat([df[numerical_cols + ['SeniorCitizen']].reset_index(drop=True), encoded_df], axis=1)
    
    # Align columns, adding missing OHE columns with 0 (essential for deployment!)
    missing_cols = set(feature_columns) - set(final_df.columns)
    for c in missing_cols:
        final_df[c] = 0
    
    # Select and order the features exactly as expected by the model
    final_features = final_df[feature_columns].values
    
    # 5. Make Prediction
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
        churn_class, churn_proba = preprocess_and_predict(user_input, model, encoder, scaler, FEATURE_COLUMNS)
        
        st.subheader("Prediction Result")

        if churn_class == 1:
            st.error(f"❌ **The customer is likely to CHURN!**")
            st.write(f"Confidence (Probability of Churn): **{churn_proba:.2f}**")
        else:
            st.success(f"✅ **The customer is NOT likely to churn.**")
            st.write(f"Confidence (Probability of No Churn): **{1 - churn_proba:.2f}**")
            
        st.balloons()
        
    except Exception as e:
        st.exception(f"An error occurred during prediction. Check the data preprocessing steps: {e}")