import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load("gmm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
columns = joblib.load("columns.pkl")

st.title("Customer Segmentation (GMM)")

st.markdown("Enter customer details to predict their segment.")

# Example fields (adjust as per your features)
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
Kidhome = st.selectbox("Number of Kids at Home", [0, 1, 2])
Teenhome = st.selectbox("Number of Teenagers at Home", [0, 1, 2])
Recency = st.slider("Days Since Last Purchase", 0, 100, 20)
MntWines = st.number_input("Amount Spent on Wine", 0, 1000, 50)
MntFruits = st.number_input("Amount Spent on Fruits", 0, 1000, 10)
MntMeatProducts = st.number_input("Amount Spent on Meat", 0, 1000, 30)
MntFishProducts = st.number_input("Amount Spent on Fish", 0, 1000, 15)
MntSweetProducts = st.number_input("Amount Spent on Sweets", 0, 1000, 5)
MntGoldProds = st.number_input("Amount Spent on Gold", 0, 1000, 20)
NumDealsPurchases = st.slider("Number of Deals Purchased", 0, 20, 2)
Married = st.selectbox("Married?", ["Yes", "No"]) == "Yes"

# Construct input
input_dict = {
    'Age': Age,
    'Income': Income,
    'Kidhome': Kidhome,
    'Teenhome': Teenhome,
    'Recency': Recency,
    'MntWines': MntWines,
    'MntFruits': MntFruits,
    'MntMeatProducts': MntMeatProducts,
    'MntFishProducts': MntFishProducts,
    'MntSweetProducts': MntSweetProducts,
    'MntGoldProds': MntGoldProds,
    'NumDealsPurchases': NumDealsPurchases,
    'Marital_Status_Married': int(Married),
}

# Add missing dummy variables as 0
for col in columns:
    if col not in input_dict:
        input_dict[col] = 0

# Build DataFrame
input_df = pd.DataFrame([input_dict])[columns]

# Predict
if st.button("Predict Segment"):
    scaled = scaler.transform(input_df)
    reduced = pca.transform(scaled)
    segment = model.predict(reduced)[0]

    segment_labels = {
        0: "Low-value customer",
        1: "High-value customer",
        2: "Mainstream customer",
        3: "Inactive customer"
        # ← Adjust these based on your analysis
    }
    label = segment_labels.get(segment, "Unknown Segment")

    st.success(f"🧾 Predicted Customer Segment: {segment} – {label}")
