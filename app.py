import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(
    page_title="Hyderabad House Rent Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load Saved Model Files
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #125ea7;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.title("🏠 Hyderabad House Rent Prediction App")
st.markdown("### Predict House Rent Instantly using Machine Learning 🤖")

st.write("---")

# Sidebar
st.sidebar.header("📊 Enter House Details")

area = st.sidebar.number_input("📐 Area (sq.ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.sidebar.number_input("🛏 Bedrooms", min_value=1, max_value=10, value=2)
washrooms = st.sidebar.number_input("🚿 Washrooms", min_value=1, max_value=10, value=2)

st.write("## 🏡 House Information")
col1, col2, col3 = st.columns(3)

col1.metric("📐 Area", f"{area} sq.ft")
col2.metric("🛏 Bedrooms", bedrooms)
col3.metric("🚿 Washrooms", washrooms)

st.write("---")

if st.button("🔮 Predict House Rent"):
    
    # Create DataFrame
    new_house = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Washrooms': [washrooms]
    })

    # Dummy Encoding Handling
    new_house = pd.get_dummies(new_house)
    new_house = new_house.reindex(columns=model_columns, fill_value=0)

    # Scaling
    new_scaled = scaler.transform(new_house)

    # Prediction
    predicted_price = model.predict(new_scaled)[0]

    st.success(f"💰 Estimated House Rent: ₹ {round(predicted_price,2):,.2f}")

    st.balloons()

st.write("---")
st.markdown("### 📈 About Model")
st.info("This model is trained using Linear Regression on Hyderabad house dataset.")
