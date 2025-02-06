import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model and scaler
fraud_model = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Streamlit UI with sidebar
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button { background-color: #ff4b4b; color: white; font-size: 20px; border-radius: 10px; }
    .stTextInput, .stNumberInput, .stSelectbox { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

# write in sidebar in markup style
st.sidebar.markdown('<h1><b>Fraud Detection System</b></h1>',unsafe_allow_html=True)
st.sidebar.markdown('<h5><b>Enter transaction details:</b></h5>', unsafe_allow_html=True)

st.markdown("""
    <h1 style="margin-bottom: 0px; margin-top: 0px"><b><a href="https://www.github.com/MetalInMyVeins/YouFraud" style="text-decoration: none; color: inherit">YouFraud</a></b><span style="font-weight: normal; font-size: 30px">    🔗</span></h1>
    <p style="margin-bottom: 2px; margin-top: 0px"><b><i>From Fraud We Guard</i></b></p>
    <hr style="margin-top: 0px; margin-bottom: 0px;">
    <hr style="margin-top: 0px; margin-bottom: 0px;">
    <hr style="margin-top: 0px; margin-bottom: 30px;">
""", unsafe_allow_html=True)

# Input fields
step = st.sidebar.number_input("**Step (1 Step = 24 Hours)**", min_value=1, step=1)
type_ = st.sidebar.selectbox("**Transaction Type**", options=["CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT", "CASH_IN"])
amount = st.sidebar.number_input("**Transaction Amount**", min_value=0.0, step=0.01)
oldbalanceOrg = st.sidebar.number_input("**Origin Account Balance Before Transaction**", min_value=0.0, step=0.01)
oldbalanceDest = st.sidebar.number_input("**Destination Account Balance Before Transaction**", min_value=0.0, step=0.01)
isFlaggedFraud = st.sidebar.selectbox("**Flagged as Fraud by Bank?**", options=[0, 1])

# Convert transaction type to numerical representation
type_mapping = {"CASH_OUT": 0, "TRANSFER": 1, "DEBIT": 2, "PAYMENT": 3, "CASH_IN": 4}
type_num = type_mapping[type_]

# Predict fraud
if st.sidebar.button("Check Fraud"):
    input_data = pd.DataFrame({
        'step': [step],
        'type': [type_num],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'oldbalanceDest': [oldbalanceDest],
        'isFlaggedFraud': [isFlaggedFraud]
    })

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Get prediction
    isFraud = fraud_model.predict(input_scaled)[0]

    # Show result
    st.subheader("Prediction Result:")
    if isFraud:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Safe.")


