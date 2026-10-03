# 💧 Smart Water Purification System - Project Context

## 📌 Project Overview
The **Smart Water Purification System** is an AI-driven decision support system designed to evaluate and predict water safety levels (**SAFE ✅**, **MODERATE ⚠️**, or **UNSAFE ❌**) using real-time telemetry inputs from 8 physical and chemical water quality sensors. 

It provides dual interactive web dashboards built with **Streamlit** and **Gradio**, permitting single-sample assessment, batch CSV predictions, exploratory data analysis, and rule-based safety threshold comparisons against standard guidelines (WHO/EPA).

---

## 📁 Repository Structure & File Map

```
water-quality-prediction/
├── app.py                          # Streamlit web application dashboard
├── gradio_app.py                   # Gradio interactive multi-tab dashboard
├── model_training.py               # ML training pipeline script (data scaling, CV, model saving)
├── ML_project.ipynb                # Jupyter Notebook for exploratory research & model experiments
├── water_purification_dataset.csv  # Primary dataset (1,200 samples, 8 features + targets)
├── model.pkl                       # Serialized trained DecisionTreeClassifier artifact
├── scaler.pkl                      # Serialized fitted StandardScaler artifact
├── requirements.txt                # Python package dependencies
├── PROJECT_CONTEXT.md              # Project context file for AI models & developers
└── README.md                       # Comprehensive user guide & setup documentation
```

---

## 📊 Dataset Schema & Features

- **Total Records**: 1,200 rows (0 missing values)
- **Input Features (8 numeric variables)**:
  1. `pH` (Float: 6.0 – 9.0): Acidity / alkalinity measure.
  2. `turbidity_NTU` (Float: 0.1 – 10.0 NTU): Water clarity indicator.
  3. `TDS_ppm` (Integer: 100 – 1000 ppm): Total Dissolved Solids.
  4. `flow_rate_L_min` (Float: 0.5 – 2.0 L/min): System flow velocity.
  5. `pressure_bar` (Float: 1.0 – 5.0 bar): Operational system pressure.
  6. `temperature_C` (Float: 15.0 – 35.0 °C): Water temperature.
  7. `usage_L_per_day` (Integer: 5 – 50 L/day): Daily system volumetric throughput.
  8. `days_since_filter_change` (Integer: 1 – 180 days): Filter element age.

- **Target Variable**: `water_quality` (Multiclass integer):
  - `0` = SAFE (Optimal quality, safe for consumption)
  - `1` = MODERATE (Pre-treatment recommended)
  - `2` = UNSAFE (Critical contamination thresholds exceeded)

---

## 🤖 ML Modeling & Serialization

- **Pre-processing**: `StandardScaler` applied across all 8 numeric features.
- **Trained Model**: `DecisionTreeClassifier` (tuned via 5-Fold `GridSearchCV`).
- **Artifacts**:
  - `model.pkl`: Pickle file containing trained classification model.
  - `scaler.pkl`: Pickle file containing trained scaler.

---

## 🚀 Execution & Command Reference

| Action | Command |
|---|---|
| Train & Save Model | `python model_training.py` |
| Launch Streamlit Dashboard | `streamlit run app.py` |
| Launch Gradio Dashboard | `python gradio_app.py` |
| Install Dependencies | `pip install -r requirements.txt` |

---

## 👤 Author Information
- **Developer**: Janvi Thakre
- **Domain**: Data Science & Machine Learning / Smart IoT Applications
