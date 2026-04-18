import streamlit as st
from google.cloud import aiplatform

# --- CONFIGURATION ---
PROJECT_ID = "neural-quarter-490713-b0"
ENDPOINT_ID = "4455696654527365120" 
LOCATION = "us-central1"

def get_prediction(instances):
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    
    # XGBoost with feature names often requires the raw 'predict' call 
    # to be formatted as a list of dictionaries.
    response = endpoint.predict(instances=instances)
    return response.predictions

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("💰 Loan Approval Prediction")

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        no_of_dependents = st.number_input("Dependents", min_value=0, value=3)
        income_annum = st.number_input("Annual Income", min_value=0, value=10000000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=200000)
        loan_term = st.number_input("Loan Term (Years)", min_value=1, value=5)
    with col2:
        cibil_score = st.number_input("CIBIL Score", min_value=300, value=700)
        education = st.selectbox("Education Status", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Are you Self Employed?", ["Yes", "No"])
        
    submitted = st.form_submit_button("Check Approval Status")

if submitted:
    # --- GUARANTEED PAYLOAD STRUCTURE ---
    # Hum wahi names use kar rahe hain jo aapke error mein dikh rahe hain.
    # Vertex AI standard container dict ko instances ki list mein expect karta hai.
    payload = {
        "no_of_dependents": float(no_of_dependents),
        "income_annum": float(income_annum),
        "loan_amount": float(loan_amount),
        "loan_term": float(loan_term),
        "cibil_score": float(cibil_score),
        "residential_assets_value": 0.0,
        "commercial_assets_value": 0.0,
        "luxury_assets_value": 0.0,
        "bank_asset_value": 0.0,
        "education_ Graduate": 1.0 if education == "Graduate" else 0.0,
        "education_ Not Graduate": 0.0 if education == "Graduate" else 1.0,
        "self_employed_ No": 0.0 if self_employed == "Yes" else 1.0,
        "self_employed_ Yes": 1.0 if self_employed == "Yes" else 0.0
    }
    
    try:
        with st.spinner('Vertex AI processing...'):
            # Double check: instances is a LIST containing the payload DICTIONARY.
            # This is technically a 2D structure of records.
            result = get_prediction([payload])
            
            score = result[0]
            if isinstance(score, list):
                score = score[0]
            
            st.divider()
            if score >= 0.5:
                st.success(f"🎉 Approved! Score: {score:.2f}")
            else:
                st.error(f"❌ Rejected! Score: {score:.2f}")
                
    except Exception as e:
        st.error(f"Error detail: {str(e)}")