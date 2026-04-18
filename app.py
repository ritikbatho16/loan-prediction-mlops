import streamlit as st
from google.cloud import aiplatform

# UI Styling
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.write("Customer ki details enter karein check karne ke liye ki loan approve hoga ya nahi.")

# --- 1. CREDENTIALS ---
PROJECT_ID = "neural-quarter-490713-b0" 
LOCATION = "us-central1"
# IMPORTANT: Naya deploy karne par ID badal jayegi, use yahan replace karein
ENDPOINT_ID = "7253557923031285760" 

# --- 2. INPUT PARAMETERS ---
st.sidebar.header("Customer Details")
no_of_dependents = st.sidebar.number_input("No of Dependents", 0, 10, 0)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income", 0, 10000000, 500000)
loan_amount = st.sidebar.number_input("Loan Amount", 0, 10000000, 100000)
loan_term = st.sidebar.number_input("Loan Term (Years)", 1, 20, 5)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 600)

if st.button("Check Eligibility"):
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)

        # --- 3. DATA PREPARATION (Order as per training dummies) ---
        edu_grad = 1.0 if education == "Graduate" else 0.0
        edu_not_grad = 0.0 if education == "Graduate" else 1.0
        emp_no = 1.0 if self_employed == "No" else 0.0
        emp_yes = 1.0 if self_employed == "Yes" else 0.0
        
        # Model expects exactly 13 floats (5 features + 4 asset placeholders + 4 dummies)
        instance = [
            float(no_of_dependents),
            float(income_annum),
            float(loan_amount),
            float(loan_term),
            float(cibil_score),
            0.0, 0.0, 0.0, 0.0, # Placeholders for asset values
            edu_grad,
            edu_not_grad,
            emp_no,
            emp_yes
        ]

        # Call Endpoint
        with st.spinner('Vertex AI se result aa raha hai...'):
            prediction = endpoint.predict(instances=[instance])
            result = prediction.predictions[0]
            
            # Score handle karna
            if isinstance(result, list): result = result[0]
            
            st.divider()
            if result >= 0.5:
                st.success(f"### Status: Loan Approved! 🎉 (Confidence: {result:.2f})")
            else:
                st.error(f"### Status: Loan Rejected ❌ (Confidence: {result:.2f})")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Check kijiye ki Endpoint ACTIVE hai aur ID sahi hai.")

st.divider()
st.info("Model: XGBoost Classifier (Plan B) | Location: Nagpur Localhost")