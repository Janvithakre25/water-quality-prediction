# 💧 Smart Water Purification & Quality Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)](https://gradio.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning solution for real-time water quality classification and purification monitoring. The system uses multi-sensor telemetry data (pH, turbidity, TDS, temperature, flow rate, pressure, daily usage, filter age) to classify water samples as **SAFE ✅**, **MODERATE ⚠️**, or **UNSAFE ❌**.

It comes equipped with **two interactive web applications** (**Streamlit** & **Gradio**), automated model training scripts, and batch processing capabilities.

---

## ✨ Features

- 🔍 **Interactive Real-Time Prediction**: Instant classification based on 8 sensor inputs with color-coded status badges and rule-based safety checks.
- 📂 **Batch CSV Processing**: Upload CSV files containing sensor logs for bulk prediction and automated export.
- 📊 **Exploratory Data Analysis (EDA)**: Interactive class distributions, correlation heatmaps, and parameter scatter plots.
- 🛡️ **WHO/EPA Safety Thresholds**: Compare pH, Turbidity, and TDS values directly against standard drinking water guidelines.
- ⚡ **Dual UI Options**: Use either **Streamlit** for a web dashboard or **Gradio** for flexible multi-tab analytics.
- 🤖 **Automated Model Pipeline**: Run `model_training.py` to evaluate multiple algorithms, tune hyperparameters via `GridSearchCV`, and output model artifacts.

---

## 📁 Project Structure

```
water-quality-prediction/
├── app.py                          # Streamlit Web Application
├── gradio_app.py                   # Gradio Interactive Dashboard
├── model_training.py               # Standalone ML training & evaluation script
├── ML_project.ipynb                # Jupyter notebook with EDA & experimental modeling
├── water_purification_dataset.csv  # Water sensor dataset (1,200 samples)
├── model.pkl                       # Saved trained DecisionTreeClassifier model
├── scaler.pkl                      # Saved fitted StandardScaler instance
├── requirements.txt                # Dependencies list
├── PROJECT_CONTEXT.md              # AI Context & technical specification file
└── README.md                       # Documentation & guide
```

---

## 📊 Dataset Overview

The dataset contains **1,200 water sensor records** across 8 physical and operational input parameters:

| Parameter | Type | Range | Description |
|---|---|---|---|
| **pH** | Float | 6.0 – 9.0 | Acidity/alkalinity level |
| **Turbidity (NTU)** | Float | 0.1 – 10.0 NTU | Cloudiness/clarity measure |
| **TDS (ppm)** | Integer | 100 – 1000 ppm | Total Dissolved Solids |
| **Flow Rate (L/min)** | Float | 0.5 – 2.0 L/min | Fluid flow velocity |
| **Pressure (bar)** | Float | 1.0 – 5.0 bar | Operational line pressure |
| **Temperature (°C)** | Float | 15.0 – 35.0 °C | Water temperature |
| **Usage (L/day)** | Integer | 5 – 50 L/day | Daily volume usage |
| **Days Since Filter Change** | Integer | 1 – 180 days | Filter element operational age |

### Target Classes (`water_quality`):
- `0` = **SAFE ✅**: Optimal parameters; safe for drinking.
- `1` = **MODERATE ⚠️**: Minor parameter deviations; check filter or pre-treat.
- `2` = **UNSAFE ❌**: Exceeds contamination threshold; immediate maintenance required.

---

## 🤖 Machine Learning Performance

Four classification algorithms were evaluated using 5-Fold Cross Validation:

| Model | Test Accuracy | 5-Fold CV Score |
|---|---|---|
| **Logistic Regression** | ~67.92% | ~67.50% |
| **AdaBoost** | ~68.75% | ~68.12% |
| **Random Forest** | ~99.58% | ~99.58% |
| **Decision Tree (Tuned)** | **99.58%** | **99.58%** |

*The Decision Tree model achieved the highest accuracy and was selected as the final production model.*

---

## 🚀 Quick Start & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Janvithakre25/water-quality-prediction.git
cd water-quality-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Model & Generate Artifacts (Optional)
```bash
python model_training.py
```
*(This creates `model.pkl` and `scaler.pkl` automatically)*

---

## 🖥️ Launching Dashboards

### Option A: Launch Gradio Interactive Dashboard
```bash
python gradio_app.py
```
*Access the Gradio web UI at `http://127.0.0.1:7860`*

### Option B: Launch Streamlit App
```bash
streamlit run app.py
```
*Access the Streamlit web UI at `http://localhost:8501`*

---

## 📝 Features & Dashboard Comparison

| Feature | Gradio App (`gradio_app.py`) | Streamlit App (`app.py`) |
|---|---|---|
| **Single Sample Telemetry** | Sliders + Probability Breakdown | Sliders + Metric Badges |
| **Batch CSV Prediction** | ✅ Upload CSV & Export CSV | ❌ |
| **EDA Visualizations** | Target Distribution & Heatmap | Countplots & Scatterplots |
| **Safe Limit Checkers** | Integrated Rule Checklist | Dedicated Tab |

---

## 🙋 Made By

**Janvi Thakre**  
*Data Science & Analytics / Ramdeobaba University*  
[GitHub Profile](https://github.com/Janvithakre25)
