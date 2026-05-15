import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD & TRAIN MODEL
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Training.csv")

    # Clean data
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('%', '')

    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)

    return df

@st.cache_resource
def train_model(df):
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:15]

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bank Churn Predictor",
    layout="wide",
    page_icon="🏦"
)

st.title("🏦 Bank Customer Churn Prediction App")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Enter Customer Details")

geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
credit_score = st.sidebar.number_input("Credit Score", 300, 850, 600)
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_crcard = st.sidebar.selectbox("Has Credit Card", [1, 0])
is_active = st.sidebar.selectbox("Is Active Member", [1, 0])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# =========================
# BUILD INPUT DATA
# =========================
def build_input():
    row = {}

    # One-hot encoding
    row['France'] = 1 if geography == 'France' else 0
    row['Germany'] = 1 if geography == 'Germany' else 0
    row['Spain'] = 1 if geography == 'Spain' else 0

    row['Male'] = 1 if gender == 'Male' else 0

    # Numeric features
    row['CreditScore'] = credit_score
    row['Age'] = age
    row['Tenure'] = tenure
    row['Balance'] = balance
    row['NumOfProducts'] = num_products
    row['HasCrCard'] = has_crcard
    row['IsActiveMember'] = is_active
    row['EstimatedSalary'] = salary

    return pd.DataFrame([row])

input_df = build_input()

# =========================
# PREDICTION
# =========================
st.subheader("Prediction")

if st.button("🔍 Predict Churn"):
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("❌ Customer will CHURN")
            else:
                st.success("✅ Customer will STAY")

        with col2:
            st.metric("Churn Probability", f"{prob:.2%}")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "#4CAF50"},
                    {'range': [40, 70], 'color': "#FFC107"},
                    {'range': [70, 100], 'color': "#F44336"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# =========================
# VISUAL ANALYSIS
# =========================
st.subheader("📊 Customer Analysis")

chart_df = pd.DataFrame({
    "Feature": ["Credit Score", "Age", "Balance", "Salary"],
    "Value": [credit_score, age, balance, salary]
})

fig = px.bar(chart_df, x="Feature", y="Value", title="Customer Profile")
st.plotly_chart(fig, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🏦 Bank Churn Prediction App | Built with Streamlit")
