import streamlit as st
import pandas as pd
import joblib

# Load saved models
model = joblib.load("gmm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
columns = joblib.load("columns.pkl")

# App title
st.title("🧠 Customer Segmentation (GMM)")
st.markdown("Enter customer details below to predict their segment.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=5000, max_value=200000, value=50000)
kidhome = st.number_input("Number of kids at home", min_value=0, max_value=5, value=0)
teenhome = st.number_input("Number of teens at home", min_value=0, max_value=5, value=0)
recency = st.number_input("Days since last purchase", min_value=0, max_value=100, value=20)

mnt_wines = st.number_input("Spending on Wine", min_value=0, max_value=1000, value=50)
mnt_fruits = st.number_input("Spending on Fruits", min_value=0, max_value=500, value=10)
mnt_meat = st.number_input("Spending on Meat", min_value=0, max_value=1000, value=30)
mnt_fish = st.number_input("Spending on Fish", min_value=0, max_value=500, value=10)
mnt_sweets = st.number_input("Spending on Sweets", min_value=0, max_value=300, value=5)
mnt_gold = st.number_input("Spending on Gold Products", min_value=0, max_value=1000, value=100)

deals = st.number_input("Number of Deals Purchased", min_value=0, max_value=20, value=2)
married = st.selectbox("Is the customer married?", options=["Yes", "No"])

# Create input DataFrame
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

input_df = pd.DataFrame([raw_input])

# Ensure all required columns exist
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[columns]

# Predict
try:
    scaled = scaler.transform(input_df)
    reduced = pca.transform(scaled)
    segment = model.predict(reduced)[0]

    st.success(f"🎯 Predicted Customer Segment: {segment}")
    
    # Optional: Interpret the segment
    segment_labels = {
        0: "Low-Value Customer",
        1: "Mid-Tier Customer",
        2: "High-Value Customer",
        3: "Inactive Customer"
    }
    st.write(f"**Segment Meaning:** {segment_labels.get(segment, 'Unknown')}")

except Exception as e:
    st.error("Prediction failed. Please check inputs or model files.")
