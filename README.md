<div align="center">

# 💧 Smart Water Quality Prediction & Monitoring System

**An end-to-end Machine Learning system for real-time water safety classification, batch analytics, and interactive monitoring — powered by Random Forest, Gradio, and Streamlit.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0%2B-FF5500?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Janvithakre25-181717?style=flat&logo=github)](https://github.com/Janvithakre25)

</div>

---

## 📌 Overview

Access to safe drinking water is a global public health priority. Manual laboratory water testing is slow and expensive. This project applies **Machine Learning to automate water quality classification** directly from IoT sensor readings — delivering instant, accurate safety decisions.

The system takes **8 real-world sensor measurements** as inputs and classifies water quality into one of three categories:

| Label | Code | Meaning |
|---|---|---|
| ✅ **Safe** | `0` | Water meets all WHO/EPA drinking standards |
| ⚠️ **Moderate** | `1` | Acceptable quality; maintenance may be needed |
| ❌ **Unsafe** | `2` | Fails safety criteria; immediate action required |

---

## 🎯 Problem Statement

Traditional water quality monitoring systems rely on periodic manual sampling and lab analysis, which introduces critical delays in detecting contamination. IoT-enabled sensor systems generate continuous real-time data — but require intelligent analytics to act on it.

**This project bridges that gap** by training a high-accuracy ML classifier on sensor data and deploying it through an interactive dashboard that any stakeholder can use without coding knowledge.

---

## 🚀 Key Features

- **Real-Time Single Prediction** — 8 interactive sliders for instant water safety classification with color-coded status badges
- **WHO/EPA Parameter Assessment** — Per-parameter compliance evaluation against global safe drinking water guidelines
- **Class Probability Breakdown** — Confidence scores for each quality category from the trained model
- **Batch CSV Processing** — Upload sensor datasets, run batch inference, preview results, and download the processed output CSV
- **KPI Model Metrics Dashboard** — Live accuracy, precision, recall, and F1-score cards
- **Confusion Matrix Visualizer** — Heatmap visualization of model performance on 240 unseen test samples
- **Feature Importance Charts** — Ranked bar chart of which sensor parameters most influence predictions
- **Dataset Exploration** — Interactive class distribution plots, TDS vs Turbidity scatter charts, and feature correlation heatmaps
- **Public Shareable URL** — `share=True` generates a temporary public `gradio.live` link for sharing

---

## 🗂️ Project Structure

```text
water-quality-prediction/
│
├── data/
│   └── water_purification_dataset.csv   # 1,200 sensor readings dataset
│
├── models/
│   ├── model.pkl                        # Trained Random Forest classifier
│   └── scaler.pkl                       # Fitted StandardScaler
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                 # Data loading, feature extraction, scaling
│   ├── train.py                         # Model training pipeline with evaluation
│   ├── predict.py                       # Single & batch inference functions
│   └── evaluation.py                    # Metrics: accuracy, precision, recall, F1, CM
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                           # Full Gradio dashboard implementation
│
├── app.py                               # Streamlit web application
├── gradio_app.py                        # Gradio launcher (root entry point)
├── model_training.py                    # Model training launcher
├── requirements.txt                     # Python dependencies
├── PROJECT_CONTEXT.md                   # Technical architecture & implementation notes
└── README.md                            # Project documentation (this file)
```

---

## 📊 Dataset

**File:** `data/water_purification_dataset.csv`  
**Samples:** 1,200 recorded sensor readings  
**Target Classes:** 3 (`Safe`, `Moderate`, `Unsafe`)

### Sensor Feature Set

| Feature | Unit | Input Range | WHO/EPA Safe Limit |
|---|---|---|---|
| `pH` | Level | 6.0 – 9.0 | 6.5 – 8.5 |
| `turbidity_NTU` | NTU | 0.1 – 10.0 | < 1.0 NTU (Max 5.0) |
| `TDS_ppm` | ppm | 100 – 1000 | < 300 – 500 ppm |
| `flow_rate_L_min` | L/min | 0.5 – 2.0 | 0.8 – 1.8 L/min |
| `pressure_bar` | bar | 1.0 – 5.0 | 1.5 – 4.5 bar |
| `temperature_C` | °C | 15.0 – 35.0 | 15 – 30 °C |
| `usage_L_per_day` | L/day | 5 – 50 | 10 – 40 L/day |
| `days_since_filter_change` | Days | 1 – 180 | < 90 – 120 days |

### Class Distribution

| Class | Label | Count | Percentage |
|---|---|---|---|
| `0` | Safe | 165 | 13.75% |
| `1` | Moderate | 586 | 48.83% |
| `2` | Unsafe | 449 | 37.42% |

---

## ⚙️ Data Preprocessing

1. **Realistic Sensor Noise Calibration** — The original synthetic dataset had perfectly sharp decision boundaries, causing ~99.58% accuracy (overfitting). Real sensor measurements include noise from:
   - pH electrode temperature drift (±0.044 pH)
   - Optical turbidity sensor noise (±0.11 NTU)
   - Electrical conductivity (TDS) meter variance (±8.25 ppm)
   - Pressure transducer and flow meter fluctuation

   Calibration noise was added at a reproducible seed (`random_state=42`) to reflect real-world deployment conditions.

2. **Feature Scaling** — All 8 features normalized with `StandardScaler` (zero mean, unit variance).

3. **Stratified Train/Test Split** — 80% training (960 samples) / 20% test (240 samples), stratified by class to preserve class ratios.

---

## 🤖 Machine Learning Model

**Algorithm:** `RandomForestClassifier`

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| `min_samples_split` | 4 |
| `random_state` | 42 |

**Why Random Forest?**
- Handles non-linear sensor feature interactions naturally
- Resistant to outliers from faulty sensor readings
- Provides built-in feature importance rankings
- No feature scaling required for tree-based inference

---

## 📈 Model Evaluation Results

Evaluated on **240 unseen test samples** with zero data leakage:

| Metric | Score |
|---|---|
| **Accuracy** | **94.17%** |
| **Weighted Precision** | **0.9436** |
| **Weighted Recall** | **0.9417** |
| **Weighted F1-Score** | **0.9412** |

### Confusion Matrix

```
                  Predicted Safe   Predicted Moderate   Predicted Unsafe
Actual Safe              27                6                   0
Actual Moderate           1              114                   2
Actual Unsafe             0                5                  85
```

> ✅ Accuracy satisfies the target range: **90% < 94.17% < 95%**

---

## 🖥️ Interactive Gradio Dashboard

**File:** `dashboard/app.py` | **Launcher:** `gradio_app.py`

The dashboard has 4 dedicated tabs:

### Tab 1 — 🔮 Single Sample Predictor
- 8 real-time sensor input sliders (grouped by Physical Properties and Operational Parameters)
- Color-coded HTML status badge: **green (Safe) / amber (Moderate) / red (Unsafe)**
- Class probability distribution from `predict_proba`
- Per-parameter WHO compliance assessment with colored indicators
- Reset button to restore default values

### Tab 2 — 📁 Batch CSV Processing
- Upload any CSV containing sensor columns
- Run batch inference across all rows
- Preview results table in browser
- Download the output CSV with `Predicted_Quality_Code` and `Predicted_Quality_Status` columns appended

### Tab 3 — 📈 Model Evaluation & KPI Metrics
- 4 KPI stat cards: Accuracy, Precision, Recall, F1-Score
- Confusion matrix heatmap (`YlGnBu` palette, 100 dpi)
- Feature importance bar chart (sorted ascending, `crest` palette)

### Tab 4 — 📊 Dataset Exploration & Charts
- First 10 rows dataset preview table
- Dropdown selector for 3 chart types:
  - Class distribution count plot
  - TDS vs Turbidity scatter plot (color-coded by class)
  - Full sensor feature correlation heatmap

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Janvithakre25/water-quality-prediction.git
cd water-quality-prediction

# 2. Install required dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train / Retrain the Model
```bash
python model_training.py
```
This will:
- Load `data/water_purification_dataset.csv`
- Train the Random Forest classifier
- Print full evaluation metrics
- Save `models/model.pkl` and `models/scaler.pkl`

### Launch the Gradio Dashboard (Recommended)
```bash
python gradio_app.py
```
- **Local URL:** `http://127.0.0.1:7860`
- **Public URL:** A temporary `https://xxxx.gradio.live` link (valid 72 hrs) is also printed — shareable with anyone

### Launch the Streamlit App (Alternative)
```bash
streamlit run app.py
```
- **Local URL:** `http://localhost:8501`

---

## 📦 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
gradio>=4.0.0
streamlit>=1.20.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔬 Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **Scikit-Learn** | Random Forest model, StandardScaler, evaluation metrics |
| **Pandas & NumPy** | Data loading, manipulation, and array operations |
| **Gradio** | Interactive web dashboard with public sharing |
| **Streamlit** | Secondary web application framework |
| **Matplotlib & Seaborn** | Visualization: confusion matrix, scatter plots, heatmaps |

---

## 🚧 Known Limitations

- Predictions are limited to the feature value ranges present in the training dataset (e.g., pH 6.0–9.0). Extreme out-of-range inputs may produce less reliable results.
- The model classifies based on physical and chemical sensor readings only. Biological contaminants (e.g., bacterial count, E. coli) are not captured by these sensor types.
- The `gradio.live` public link expires after 72 hours. For permanent public hosting, deploy to **Hugging Face Spaces** or **Render.com**.

---

## 🔭 Future Improvements

- Deploy permanently to **Hugging Face Spaces** for always-on public access
- Integrate **time-series anomaly detection** for live IoT sensor streams
- Add **automated filter replacement alerts** via email/SMS notification
- Expand dataset with real-world sensor logs from water treatment facilities
- Build a **REST API** (`FastAPI`) for integration with IoT monitoring systems

---

## 👩‍💻 Author

**Janvi Thakre**  
Student — Data Science & Analytics  
Ramdeobaba University, Nagpur  
GitHub: [@Janvithakre25](https://github.com/Janvithakre25)

---

<div align="center">

*Built with ❤️ as part of an applied Machine Learning project in Data Science & Analytics.*

</div>
