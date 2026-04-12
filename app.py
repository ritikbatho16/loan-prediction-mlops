import streamlit as st
from google.cloud import aiplatform

# UI Styling - Loan Approval ke hisaab se
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.write("Customer ki details enter karein check karne ke liye ki loan approve hoga ya nahi.")

# --- 1. CREDENTIALS REPLACEMENT ---
PROJECT_ID = "neural-quarter-490713-b0" 
# Endpoint ID aapko GCP Console > Vertex AI > Endpoints mein jaakar milegi jab model deploy ho jayega
ENDPOINT_ID = "your-endpoint-id-here" 

# --- 2. INPUT PARAMETERS (Loan Dataset ke hisaab se) ---
st.sidebar.header("Customer Details")
no_of_dependents = st.sidebar.number_input("No of Dependents", 0, 10, 0)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income", 0, 10000000, 500000)
loan_amount = st.sidebar.number_input("Loan Amount", 0, 10000000, 100000)
loan_term = st.sidebar.number_input("Loan Term (Years)", 1, 20, 5)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 600)

# Prediction Logic
if st.button("Check Eligibility"):
    try:
        aiplatform.init(project=PROJECT_ID, location='us-central1')
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)

        # --- 3. DATA PREPARATION ---
        # Note: Aapko wahi features bhejne honge jo train.py mein the.
        # Agar aapne train.py mein pd.get_dummies use kiya hai, 
        # toh yaha bhi categorical values ko numbers (0/1) mein convert karna hoga.
        
        edu_val = 1 if education == "Graduate" else 0
        emp_val = 1 if self_employed == "Yes" else 0
        
        # Example instance (Order wahi rakhein jo training ke waqt tha)
        instance = [
            no_of_dependents, edu_val, emp_val, income_annum, 
            loan_amount, loan_term, cibil_score
        ]

        prediction = endpoint.predict(instances=[instance])
        
        # Result logic (0 = Rejected, 1 = Approved)
        result = prediction.predictions[0]
        
        if result >= 0.5:
            st.success("### Status: Loan Approved! 🎉")
        else:
            st.error("### Status: Loan Rejected ❌")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure your Endpoint ID is correct and the endpoint is ACTIVE.")

st.divider()
st.info("Model: XGBoost Classifier | Data: Loan Approval Dataset")

#ritik