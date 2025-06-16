import streamlit as st
import pickle
import numpy as np

# Load the trained GMM model
with open("gmm_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Segmentation using GMM")

st.markdown("Enter customer details to predict their segment")

# Input fields - match these to what your GMM model was trained on
age = st.slider("Age", 18, 70, 30)
income = st.number_input("Annual Income (in $1000s)", min_value=10, max_value=150, value=40)
score = st.slider("Spending Score (1-100)", 1, 100, 50)

# Predict Button
if st.button("Predict Segment"):
    # Input data format as expected by GMM
    input_data = np.array([[age, income, score]])
    
    # Predict the cluster
    cluster = model.predict(input_data)[0]
    
    st.success(f"The customer belongs to Segment: {cluster}")
