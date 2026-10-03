import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Smart Water Purification System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
@st.cache_resource
def load_ml_resources():
    model, scaler = None, None
    if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
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

model, scaler = load_ml_resources()
df = load_dataset()

# -----------------------------
# HEADER & SIDEBAR NAVIGATION
# -----------------------------
st.title("💧 Smart Water Purification & Quality System")
st.markdown("Predict water safety, analyze sensor metrics, and evaluate filter maintenance requirements.")

if model is None or scaler is None:
    st.error("⚠️ `model.pkl` or `scaler.pkl` missing! Please run `python model_training.py` first.")
    st.stop()

menu = st.sidebar.radio(
    "📌 Navigation Menu",
    ["🔮 Prediction", "📊 Dataset Insights", "📈 Model Performance", "📘 WHO Standards"]
)

LABEL_MAP = {0: "Safe ✅", 1: "Moderate ⚠️", 2: "Unsafe ❌"}

# =============================
# 1. PREDICTION TAB
# =============================
if menu == "🔮 Prediction":
    st.header("🎛️ Sensor Input Parameters")
    
    col1, col2 = st.columns(2)

    with col1:
        pH = st.slider("pH Level", 6.0, 9.0, 7.2, 0.01, help="Acidity/alkalinity level (Optimal: 6.5 - 8.5)")
        turbidity = st.slider("Turbidity (NTU)", 0.1, 10.0, 2.0, 0.1, help="Water clarity (Safe: < 1.0 - 5.0 NTU)")
        tds = st.slider("TDS (ppm)", 100, 1000, 300, 10, help="Total Dissolved Solids (Ideal: < 300-500 ppm)")
        flow = st.slider("Flow Rate (L/min)", 0.5, 2.0, 1.0, 0.05, help="Filtration flow velocity")

    with col2:
        pressure = st.slider("Pressure (bar)", 1.0, 5.0, 2.5, 0.1, help="Operating pressure")
        temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0, 0.5, help="Water temperature")
        usage = st.slider("Daily Usage (L/day)", 5, 50, 20, 1, help="Volume consumed daily")
        days = st.slider("Days Since Filter Change", 1, 180, 60, 1, help="Days since last filter service")

    st.markdown("---")

    if st.button("⚡ Evaluate Water Quality", use_container_width=True):
        input_data = np.array([[pH, turbidity, tds, flow, pressure, temp, usage, days]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 0:
                st.success("### Status: SAFE ✅")
                st.info("Water meets quality standards for consumption.")
            elif prediction == 1:
                st.warning("### Status: MODERATE ⚠️")
                st.warning("Acceptable quality, but filtration maintenance may be required soon.")
            else:
                st.error("### Status: UNSAFE ❌")
                st.error("Water fails safety criteria. Filtration or filter change mandatory!")

        with res_col2:
            st.markdown("### 📋 Parameter Evaluation")
            evals = []
            if 6.5 <= pH <= 8.5:
                evals.append("✅ **pH**: Within safe range (6.5 - 8.5)")
            else:
                evals.append("❌ **pH**: Outside recommended safe limits (6.5 - 8.5)")

            if turbidity <= 1.0:
                evals.append("✅ **Turbidity**: Excellent clarity (< 1.0 NTU)")
            elif turbidity <= 5.0:
                evals.append("⚠️ **Turbidity**: Acceptable limit (< 5.0 NTU)")
            else:
                evals.append("❌ **Turbidity**: High cloudiness (> 5.0 NTU)")

            if tds <= 300:
                evals.append("✅ **TDS**: Excellent low mineral content (< 300 ppm)")
            elif tds <= 500:
                evals.append("⚠️ **TDS**: Fair mineral content (300 - 500 ppm)")
            else:
                evals.append("❌ **TDS**: High dissolved solids (> 500 ppm)")

            if days > 90:
                evals.append("⚠️ **Filter Service**: Filter active for over 90 days. Service recommended.")
            else:
                evals.append("✅ **Filter Service**: Filter age within normal operating range.")

            for ev in evals:
                st.write(ev)

# =============================
# 2. DATASET INSIGHTS TAB
# =============================
elif menu == "📊 Dataset Insights":
    st.header("📊 Dataset Exploratory Analysis")

    if df is not None:
        st.write(f"Dataset contains **{len(df)}** recorded samples.")
        
        tab1, tab2, tab3 = st.tabs(["Class Distribution", "TDS vs Turbidity", "Correlation Heatmap"])

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            sns.countplot(x='water_quality', data=df, palette=["#2ea44f", "#d97706", "#dc2626"], ax=ax1)
            ax1.set_xticklabels(["Safe (0)", "Moderate (1)", "Unsafe (2)"])
            ax1.set_title("Water Quality Class Distribution")
            st.pyplot(fig1)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            sns.scatterplot(x='TDS_ppm', y='turbidity_NTU', hue='water_quality', palette=["#2ea44f", "#d97706", "#dc2626"], data=df, ax=ax2)
            ax2.set_title("TDS vs Turbidity grouped by Water Quality")
            st.pyplot(fig2)

        with tab3:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            num_df = df.select_dtypes(include=[np.number])
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax3)
            ax3.set_title("Feature Correlation Matrix")
            st.pyplot(fig3)
    else:
        st.error("`water_purification_dataset.csv` not found.")

# =============================
# 3. MODEL PERFORMANCE TAB
# =============================
elif menu == "📈 Model Performance":
    st.header("📈 Model Validation & Metrics")

    if df is not None:
        X = df[["pH", "turbidity_NTU", "TDS_ppm", "flow_rate_L_min", "pressure_bar", "temperature_C", "usage_L_per_day", "days_since_filter_change"]]
        y = df["water_quality"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Safe", "Moderate", "Unsafe"], yticklabels=["Safe", "Moderate", "Unsafe"], ax=ax)
            ax.set_ylabel("Actual Class")
            ax.set_xlabel("Predicted Class")
            st.pyplot(fig)

        with col_b:
            st.subheader("Feature Importance")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                sns.barplot(x=importances, y=X.columns, palette="viridis", ax=ax_imp)
                ax_imp.set_title("Feature Relative Importance")
                st.pyplot(fig_imp)
            else:
                st.info("Selected model does not support tree feature importances.")
    else:
        st.error("Dataset unavailable to compute performance matrix.")

# =============================
# 4. WHO STANDARDS TAB
# =============================
elif menu == "📘 WHO Standards":
    st.header("📘 Safe Drinking Water Quality Guidelines")
    st.markdown(
        """
        - **pH (6.5 – 8.5)**: Measures hydrogen-ion concentration. Extreme pH values irritate skin/eyes and corrode pipes.
        - **Turbidity (< 1.0 – 5.0 NTU)**: Suspended solids in water. Higher values shield bacteria from disinfection.
        - **TDS (< 300 – 500 ppm)**: Total minerals, salts, and metals dissolved in water.
        - **Filter Lifecycle (< 90-120 days)**: Standard lifespan before biofilm growth or membrane saturation occurs.
        """
    )