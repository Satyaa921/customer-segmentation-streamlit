import streamlit as st
import numpy as np
import pickle

# Load pre-trained model components
with open("gmm_model.pkl", "rb") as f:
    gmm = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Set up UI
st.title("Customer Segmentation using GMM")
st.markdown("Enter customer profile details to find their segment")

# Define input fields
income = st.number_input("Annual Income (₹)", min_value=5000, max_value=1000000, value=50000, step=5000)
recency = st.slider("Recency (Days since last purchase)", 0, 100, 30)
mnt_wines = st.number_input("Amount spent on Wines", 0, 2000, 100)
mnt_fruits = st.number_input("Amount spent on Fruits", 0, 500, 20)
mnt_meat = st.number_input("Amount spent on Meat Products", 0, 2000, 150)
mnt_fish = st.number_input("Amount spent on Fish Products", 0, 1000, 50)
mnt_sweets = st.number_input("Amount spent on Sweet Products", 0, 500, 20)
mnt_gold = st.number_input("Amount spent on Gold Products", 0, 500, 50)
num_deals = st.slider("Number of Deals Purchased", 0, 15, 3)
num_web = st.slider("Web Purchases", 0, 20, 4)
num_catalog = st.slider("Catalog Purchases", 0, 20, 2)
num_store = st.slider("Store Purchases", 0, 20, 5)
num_visits = st.slider("Website Visits in Last Month", 0, 20, 5)
age = st.slider("Age", 18, 90, 35)

if st.button("Predict Segment"):
    # Assemble input vector
    input_data = np.array([[income, recency, mnt_wines, mnt_fruits, mnt_meat,
                            mnt_fish, mnt_sweets, mnt_gold, num_deals,
                            num_web, num_catalog, num_store, num_visits, age]])
    
    # Preprocessing: Scale → PCA → Predict
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    segment = gmm.predict(input_pca)[0]

    # Optional: Label segments if known (customize as per your data insights)
    segment_labels = {
        0: "Segment 0 - Budget Buyers",
        1: "Segment 1 - Premium Shoppers",
        2: "Segment 2 - Occasional Shoppers",
        3: "Segment 3 - Active Loyal Customers"
    }

    st.success(f"The customer belongs to: {segment_labels.get(segment, f'Segment {segment}')}")
