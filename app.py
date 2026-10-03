import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Smart Water Purification System",
    page_icon="💧",
    layout="wide"
)

# -----------------------------
# RESOURCE CACHING
# -----------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
        return None, None
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_dataset():
    if os.path.exists("water_purification_dataset.csv"):
        return pd.read_csv("water_purification_dataset.csv")
    return None

model, scaler = load_artifacts()
df = load_dataset()

# -----------------------------
# HEADER & SIDEBAR
# -----------------------------
st.title("💧 Smart Water Purification & Quality Prediction System")
st.markdown("""
This intelligent decision support app utilizes Machine Learning to classify water quality into 
**SAFE ✅**, **MODERATE ⚠️**, or **UNSAFE ❌** based on real-time multi-sensor telemetry readings.
""")

if model is None or scaler is None:
    st.error("⚠️ Model (`model.pkl`) or Scaler (`scaler.pkl`) not found! Please run `python model_training.py` first.")
    st.stop()

menu = st.sidebar.selectbox(
    "Navigation Menu",
    ["Prediction", "Graphs & EDA", "User Analysis & Safe Limits", "Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset Features**:
- pH (6.0 - 9.0)
- Turbidity (0.1 - 10.0 NTU)
- TDS (100 - 1000 ppm)
- Flow Rate (0.5 - 2.0 L/min)
- Pressure (1.0 - 5.0 bar)
- Temperature (15.0 - 35.0 °C)
- Usage (5 - 50 L/day)
- Days Since Filter Change (1 - 180)
""")

# =============================
# 1. PREDICTION
# =============================
if menu == "Prediction":
    st.header("🔍 Interactive Sensor Telemetry Prediction")
    st.write("Adjust the parameters below to assess real-time water purification safety.")

    col1, col2 = st.columns(2)

    with col1:
        pH = st.slider("pH Level", 6.0, 9.0, 7.2, step=0.01)
        turbidity = st.slider("Turbidity (NTU)", 0.1, 10.0, 2.5, step=0.1)
        tds = st.slider("TDS (ppm)", 100, 1000, 350, step=5)
        flow = st.slider("Flow Rate (L/min)", 0.5, 2.0, 1.2, step=0.05)

    with col2:
        pressure = st.slider("Pressure (bar)", 1.0, 5.0, 2.8, step=0.1)
        temp = st.slider("Temperature (°C)", 15.0, 35.0, 24.0, step=0.5)
        usage = st.slider("Daily Usage (L/day)", 5, 50, 25, step=1)
        days = st.slider("Days Since Filter Change", 1, 180, 45, step=1)

    if st.button("🚀 Analyze Water Quality", type="primary"):
        input_data = np.array([[pH, turbidity, tds, flow, pressure, temp, usage, days]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        st.subheader("Result Summary")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("pH Level", f"{pH}")
        with m_col2:
            st.metric("TDS Level", f"{tds} ppm")
        with m_col3:
            st.metric("Filter Age", f"{days} days")

        if prediction == 0:
            st.success("### Water Quality Status: SAFE ✅")
            st.info("The water parameter readings fall within safe consumption standards.")
        elif prediction == 1:
            st.warning("### Water Quality Status: MODERATE ⚠️")
            st.write("Parameters indicate moderate contamination or filter degradation. Pre-treatment or filter check recommended.")
        else:
            st.error("### Water Quality Status: UNSAFE ❌")
            st.write("Water parameters exceed safe threshold limits! Immediate filtration servicing or maintenance required.")

# =============================
# 2. GRAPHS & EDA
# =============================
elif menu == "Graphs & EDA":
    st.header("📊 Dataset Exploratory Analysis")

    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Water Quality Class Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(x='water_quality', data=df, palette='viridis', ax=ax1)
            ax1.set_xticklabels(["Safe (0)", "Moderate (1)", "Unsafe (2)"])
            ax1.set_ylabel("Sample Count")
            st.pyplot(fig1)

        with col2:
            st.subheader("TDS vs Turbidity Relationship")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='TDS_ppm', y='turbidity_NTU', hue='water_quality', data=df, palette='Set2', ax=ax2)
            ax2.set_xlabel("TDS (ppm)")
            ax2.set_ylabel("Turbidity (NTU)")
            st.pyplot(fig2)

        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("Dataset not loaded.")

# =============================
# 3. USER ANALYSIS & SAFE LIMITS
# =============================
elif menu == "User Analysis & Safe Limits":
    st.header("🔍 Safe Limit Guidelines & Threshold Analysis")
    st.write("Compare specific key chemical/physical parameters against standard safe threshold guidelines.")

    col1, col2, col3 = st.columns(3)
    with col1:
        u_ph = st.number_input("pH Reading", 0.0, 14.0, 7.0)
    with col2:
        u_turb = st.number_input("Turbidity (NTU)", 0.0, 20.0, 2.0)
    with col3:
        u_tds = st.number_input("TDS (ppm)", 0, 2000, 300)

    st.markdown("### Safety Threshold Status")
    
    # pH check
    if 6.5 <= u_ph <= 8.5:
        st.success(f"**pH ({u_ph})**: Optimal (Standard: 6.5 - 8.5)")
    else:
        st.error(f"**pH ({u_ph})**: Out of Range! (Standard: 6.5 - 8.5)")

    # Turbidity check
    if u_turb <= 5.0:
        st.success(f"**Turbidity ({u_turb} NTU)**: Acceptable (Standard: < 5.0 NTU)")
    else:
        st.warning(f"**Turbidity ({u_turb} NTU)**: High (Standard: < 5.0 NTU)")

    # TDS check
    if u_tds <= 500:
        st.success(f"**TDS ({u_tds} ppm)**: Good (Standard: < 500 ppm)")
    elif u_tds <= 1000:
        st.warning(f"**TDS ({u_tds} ppm)**: Fair / High Dissolved Solids (Standard: < 500 ppm)")
    else:
        st.error(f"**TDS ({u_tds} ppm)**: Excessive Dissolved Solids!")

# =============================
# 4. MODEL PERFORMANCE
# =============================
elif menu == "Model Performance":
    st.header("📈 Model Diagnostics & Confusion Matrix")

    if df is not None:
        X = df.drop(columns=["water_quality", "filter_replacement", "maintenance_required"], errors="ignore")
        y = df["water_quality"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred)
        labels = ["Safe (0)", "Moderate (1)", "Unsafe (2)"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col2:
            st.subheader("Performance Metrics")
            report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
    else:
        st.warning("Dataset not loaded.")