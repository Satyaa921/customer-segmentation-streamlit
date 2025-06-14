
import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and feature names
gmm = joblib.load("gmm_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

st.set_page_config(page_title="Customer Segmentation with GMM", layout="centered")

st.title("🧠 Customer Segmentation App (GMM Clustering)")
st.write("This app predicts the customer segment using Gaussian Mixture Model (GMM) clustering.")

# Input form for user data
with st.form("customer_input_form"):
    inputs = []
    for feature in feature_names:
        value = st.number_input(f"Enter {feature}:", format="%.2f")
        inputs.append(value)

    submitted = st.form_submit_button("Predict Segment")

if submitted:
    # Convert inputs to array, scale, and predict
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = gmm.predict(input_scaled)[0]

    st.success(f"🔍 Predicted Customer Segment: **Cluster {prediction}**")

    # Optional: Cluster info
    st.info("Note: This segmentation is unsupervised. The clusters do not have predefined labels.")
