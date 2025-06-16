
import streamlit as st
import numpy as np
import joblib

# Load trained components
gmm = joblib.load(open("gmm_model.pkl", "rb"))
scaler = joblib.load(open("scaler.pkl", "rb"))
pca = joblib.load(open("pca.pkl", "rb"))
columns = joblib.load(open("columns.pkl", "rb"))  # list of ordered feature names

st.title("Customer Segmentation using GMM")
st.markdown("Provide customer details to predict the correct segment.")

# Input fields stored in dictionary
inputs = {}

inputs["Income"] = st.number_input("Annual Income (₹)", min_value=5000, max_value=1000000, value=50000, step=5000)
inputs["Recency"] = st.slider("Recency (Days since last purchase)", 0, 100, 30)
inputs["MntWines"] = st.number_input("Amount spent on Wines", 0, 2000, 100)
inputs["MntFruits"] = st.number_input("Amount spent on Fruits", 0, 500, 20)
inputs["MntMeatProducts"] = st.number_input("Amount spent on Meat Products", 0, 2000, 150)
inputs["MntFishProducts"] = st.number_input("Amount spent on Fish Products", 0, 1000, 50)
inputs["MntSweetProducts"] = st.number_input("Amount spent on Sweet Products", 0, 500, 20)
inputs["MntGoldProds"] = st.number_input("Amount spent on Gold Products", 0, 500, 50)
inputs["NumDealsPurchases"] = st.slider("Number of Deals Purchased", 0, 15, 3)
inputs["NumWebPurchases"] = st.slider("Web Purchases", 0, 20, 4)
inputs["NumCatalogPurchases"] = st.slider("Catalog Purchases", 0, 20, 2)
inputs["NumStorePurchases"] = st.slider("Store Purchases", 0, 20, 5)
inputs["NumWebVisitsMonth"] = st.slider("Website Visits in Last Month", 0, 20, 5)
inputs["Age"] = st.slider("Age", 18, 90, 35)

if st.button("Predict Segment"):
    # Create ordered input vector
    input_vector = np.array([[inputs[col] for col in columns]])

    # Preprocess: scale -> PCA -> predict
    scaled = scaler.transform(input_vector)
    reduced = pca.transform(scaled)
    segment = gmm.predict(reduced)[0]

    # Correct segment interpretations based on real data
labels = {
    0: "Premium Frequent Shoppers",
    1: "Loyal High-Spending Customers",
    2: "Budget Buyers",
    3: "Moderate Shoppers"}

    st.success(f"The customer belongs to: {labels.get(segment, f'Segment {segment}')}")
