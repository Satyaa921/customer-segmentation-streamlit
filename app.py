import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved models and column structure
model = joblib.load("gmm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Customer Segmentation (GMM)", layout="centered")
st.title("Customer Segmentation")
st.markdown("Enter customer details to predict their segment using the GMM model.")

# Collect input features from user
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
Income = st.number_input("Income", min_value=0, max_value=200000, value=60000)
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

# Collect raw input
raw_input = {
    "Age": age,
    "Income": income,
    "Kidhome": kidhome,
    "Teenhome": teenhome,
    "Recency": recency,
    "MntWines": mnt_wines,
    "MntFruits": mnt_fruits,
    "MntMeatProducts": mnt_meat,
    "MntFishProducts": mnt_fish,
    "MntSweetProducts": mnt_sweets,
    "MntGoldProds": mnt_gold,
    "NumDealsPurchases": deals,
    "Marital_Status_Married": 1 if married == "Yes" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([raw_input])

# Ensure all expected columns are present
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with default 0

# Reorder columns to match training
input_df = input_df[columns]

# Predict segment
if st.button("Predict Segment"):
    try:
        # Step 1: Scale input
        scaled = scaler.transform(input_df)

        # Step 2: Apply PCA
        reduced = pca.transform(scaled)

        # Step 3: Predict with GMM
        if reduced.shape[1] != model.means_.shape[1]:
            st.error("Mismatch in PCA components and model input size.")
        else:
            segment = model.predict(reduced)[0]

            # Map segment to label (edit based on analysis)
            segment_labels = {
                0: "Low-value customer",
                1: "High-value customer",
                2: "Mainstream customer",
                3: "Inactive customer"
            }
            label = segment_labels.get(segment, "Unknown Segment")
            st.success(f"🧾 Predicted Customer Segment: {segment} – {label}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
