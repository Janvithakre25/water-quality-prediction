import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.preprocessing import FEATURE_COLS, load_data, load_scaler, split_and_scale_data
from src.predict import predict_single_sample, load_trained_model, LABEL_MAP
from src.evaluation import evaluate_model_performance

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Smart Water Purification System",
    page_icon="💧",
    layout="wide"
)

model = load_trained_model()
scaler = load_scaler()
try:
    df = load_data()
except Exception:
    df = None

st.title("💧 Smart Water Quality Prediction System")
st.markdown("Real-time water quality evaluation, parameter assessment, and model performance metrics.")

if model is None or scaler is None:
    st.error("Model or Scaler file missing! Run `python model_training.py` first.")
    st.stop()

menu = st.sidebar.radio("Navigation", ["🔮 Single Prediction", "📊 Dataset Analytics", "📈 Model Metrics"])

if menu == "🔮 Single Prediction":
    st.header("🎛️ Sensor Inputs")
    col1, col2 = st.columns(2)

    with col1:
        pH = st.slider("pH Level", 6.0, 9.0, 7.2, 0.01)
        turbidity = st.slider("Turbidity (NTU)", 0.1, 10.0, 2.0, 0.1)
        tds = st.slider("TDS (ppm)", 100, 1000, 300, 10)
        flow = st.slider("Flow Rate (L/min)", 0.5, 2.0, 1.0, 0.05)

    with col2:
        pressure = st.slider("Pressure (bar)", 1.0, 5.0, 2.5, 0.1)
        temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
        usage = st.slider("Usage (L/day)", 5, 50, 20, 1)
        days = st.slider("Days Since Filter Change", 1, 180, 60, 1)

    if st.button("⚡ Predict Water Quality", use_container_width=True):
        sample_dict = {
            "pH": pH, "turbidity_NTU": turbidity, "TDS_ppm": tds,
            "flow_rate_L_min": flow, "pressure_bar": pressure,
            "temperature_C": temp, "usage_L_per_day": usage,
            "days_since_filter_change": days
        }
        pred_code, status_label, probs = predict_single_sample(sample_dict, model, scaler)

        if pred_code == 0:
            st.success(f"### Result: {status_label}")
        elif pred_code == 1:
            st.warning(f"### Result: {status_label}")
        else:
            st.error(f"### Result: {status_label}")

elif menu == "📊 Dataset Analytics":
    st.header("📊 Dataset Exploratory Visualizations")
    if df is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=df, x="water_quality", palette=["#2ea44f", "#d97706", "#dc2626"], ax=ax)
        ax.set_xticklabels(["Safe (0)", "Moderate (1)", "Unsafe (2)"])
        ax.set_title("Water Quality Distribution")
        st.pyplot(fig)
    else:
        st.error("Dataset file not available.")

elif menu == "📈 Model Metrics":
    st.header("📈 Model Evaluation Results (Tuned Decision Tree)")
    st.markdown(
        """
        - **Accuracy**: **97.08%**
        - **Precision**: **0.9711**
        - **Recall**: **0.9708**
        - **F1-Score**: **0.9708**
        """
    )
    if df is not None:
        X = df[FEATURE_COLS]
        y = df["water_quality"]
        _, X_test, _, _, _, y_test, _ = split_and_scale_data(X, y)
        metrics = evaluate_model_performance(model, X_test, y_test)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Safe", "Moderate", "Unsafe"],
                    yticklabels=["Safe", "Moderate", "Unsafe"], ax=ax)
        ax.set_title("Confusion Matrix (240 Test Samples)")
        st.pyplot(fig)