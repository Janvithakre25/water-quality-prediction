import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except:
    st.error("Model or scaler file not found!")
    st.stop()

# -----------------------------
# LOAD DATA (SAFE)
# -----------------------------
try:
    df = pd.read_csv("water_purification_dataset.csv")
except:
    st.warning("Dataset not found → Graphs disabled")
    df = None

# -----------------------------
# UI START
# -----------------------------
st.title("💧 Smart Water Purification System")
st.write("App Loaded Successfully ✅")

menu = st.sidebar.selectbox(
    "Menu",
    ["Prediction", "Graphs", "Model Performance"]
)

# =============================
# 1. PREDICTION
# =============================
if menu == "Prediction":

    st.header("Enter Sensor Values")

    pH = st.slider("pH", 6.0, 9.0, 7.0)
    turbidity = st.slider("Turbidity (NTU)", 0.1, 10.0, 2.0)
    tds = st.slider("TDS (ppm)", 100, 1000, 300)
    flow = st.slider("Flow Rate (L/min)", 0.5, 2.0, 1.0)
    pressure = st.slider("Pressure (bar)", 1.0, 5.0, 2.5)
    temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0)
    usage = st.slider("Usage (L/day)", 5, 50, 20)
    days = st.slider("Days Since Filter Change", 1, 180, 60)

    if st.button("Predict"):

        input_data = np.array([[pH, turbidity, tds, flow, pressure, temp, usage, days]])
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        if prediction == 0:
            st.success("Water Quality: SAFE ✅")
        elif prediction == 1:
            st.warning("Water Quality: MODERATE ⚠️")
        else:
            st.error("Water Quality: UNSAFE ❌")

# =============================
# 2. GRAPHS (SAFE)
# =============================
elif menu == "Graphs":

    st.header("📊 Key Insights")

    if df is not None:

        # Count Plot
        fig1, ax1 = plt.subplots()
        sns.countplot(x='water_quality', data=df, ax=ax1)
        st.pyplot(fig1)

        # Scatter Plot
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='TDS_ppm', y='turbidity_NTU',
                        hue='water_quality', data=df, ax=ax2)
        st.pyplot(fig2)

    else:
        st.warning("Dataset not available")

# =============================
# 3. MODEL PERFORMANCE
# =============================
elif menu == "Model Performance":

    st.header("📈 Confusion Matrix")

    if df is not None:

        X = df.drop(columns=["water_quality", "filter_replacement", "maintenance_required"])
        y = df["water_quality"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        labels = ["Safe", "Moderate", "Unsafe"]

        fig3, ax3 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels,
                    ax=ax3)

        st.pyplot(fig3)

    else:
        st.warning("Dataset not available")