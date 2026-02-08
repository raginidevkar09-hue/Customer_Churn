import streamlit as st
import requests

# ================= CONFIG =================
BACKEND_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
/* Full page background image */
.stApp {
    background: url("https://images.unsplash.com/photo-1557683316-973673baf926");
    background-size: cover;
    background-attachment: fixed;
}

/* Overlay to improve readability */
.main::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(15, 23, 42, 0.85);
    z-index: -1;
}

/* Cards */
.card {
    background-color: rgba(30, 41, 59, 0.92);
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
}

/* Text */
h1, h2, h3, label, p {
    color: #f8fafc !important;
}

.success { color: #22c55e; }
.error { color: #ef4444; }
.metric { color: #38bdf8; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1>📊 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.markdown("Predict whether a customer will leave the bank using Machine Learning.")

# ================= LAYOUT =================
col1, col2 = st.columns(2)

# ================= TRAIN MODEL =================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔧 Train Model")

    train_file = st.file_uploader("Upload Training CSV", type=["csv"], key="train")

    if st.button("🚀 Train Model"):
        if train_file:
            with st.spinner("Training model..."):
                response = requests.post(
                    f"{BACKEND_URL}/train",
                    files={"file": train_file}
                )

                if response.status_code == 200:
                    data = response.json()

                    st.markdown(f"""
                    <div class="output-box">
                        <p class="output-text">✅ Model Trained Successfully</p>
                        <p class="output-text">📊 Accuracy: {data['accuracy']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown("<div class='error'>Training Failed</div>", unsafe_allow_html=True)
        else:
            st.warning("Please upload training CSV")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= TEST MODEL =================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧪 Test Model")

    test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test")

    if st.button("📊 Test Model"):
        if test_file:
            with st.spinner("Testing model..."):
                response = requests.post(
                    f"{BACKEND_URL}/test",
                    files={"file": test_file}
                )

                if response.status_code == 200:
                    data = response.json()

                    st.markdown(f"""
                    <div class="output-box">
                        <p class="output-text">🧪 Test Accuracy: {data['test_accuracy']}</p>
                        <p class="output-text">📁 Total Records: {data['total_records']}</p>
                        <p class="output-text">⚠️ Predicted Churn: {data['predicted_churn']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown("<div class='error'>Testing Failed</div>", unsafe_allow_html=True)
        else:
            st.warning("Please upload test CSV")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Predict Customer Churn")

col3, col4, col5 = st.columns(3)

with col3:
    age = st.number_input("Age", 18, 100, 35)
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    tenure = st.number_input("Tenure (Years)", 0, 10, 3)

with col4:
    balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

with col5:
    gender = st.selectbox("Gender", ["Male", "Female"])
    is_active = st.selectbox("Is Active Member", [0, 1])

if st.button("🔍 Predict Churn"):
    payload = {
        "Age": age,
        "CreditScore": credit_score,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "Gender": gender,
        "IsActiveMember": is_active
    }

    response = requests.post(f"{BACKEND_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()

        if result["churn_prediction"] == 1:
            st.error(f"⚠️ Customer is likely to churn (Probability: {result['churn_probability']})")
        else:
            st.success(f"✅ Customer is NOT likely to churn (Probability: {result['churn_probability']})")
    else:
        st.error("Prediction failed")

st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown("Built with ❤️ using FastAPI + Streamlit + ML")
