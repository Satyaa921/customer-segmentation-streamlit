import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
columns = joblib.load("columns.pkl")

st.title("Customer Segmentation App")
st.markdown("Upload customer data (CSV) to classify them into segments.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(user_df.head())

    # Preprocessing
    df = pd.get_dummies(user_df, drop_first=True)
    for col in columns:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns
    df = df[columns]  # Reorder to match training

    # Scaling and PCA
    scaled = scaler.transform(df)
    reduced = pca.transform(scaled)

    # Prediction
    clusters = model.predict(reduced)
    user_df["Segment"] = clusters

    st.subheader("Segmentation Results")
    st.dataframe(user_df[["Segment"]])

    st.success(" Segmentation complete!")
