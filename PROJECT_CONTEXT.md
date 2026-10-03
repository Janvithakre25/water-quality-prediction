# 💧 Project Context: Smart Water Purification & Quality Prediction System

## 📌 Project Overview
The **Smart Water Purification System** is an end-to-end machine learning project designed to evaluate water safety and predict water quality categories based on IoT sensor inputs. The system classifies water quality into three distinct levels (**Safe**, **Moderate**, **Unsafe**) and provides actionable insights for filtration and system maintenance.

---

## 🏗️ System Architecture & File Structure

```
water-quality-prediction/
│
├── PROJECT_CONTEXT.md              # Detailed technical context, architecture, & parameters
├── README.md                       # Comprehensive user guide, setup instructions, & overview
├── requirements.txt                # Python package dependencies
├── water_purification_dataset.csv  # Raw sensor readings & ground truth dataset
│
├── model_training.py               # Machine learning pipeline (EDA, training, tuning, saving)
├── model.pkl                       # Serialized trained machine learning model (Random Forest / Best classifier)
├── scaler.pkl                      # Serialized StandardScaler instance
│
├── app.py                          # Streamlit web application dashboard
└── gradio_app.py                   # Interactive Gradio dashboard & batch prediction UI
```

---

## 📊 Dataset Features & Specifications

The dataset `water_purification_dataset.csv` consists of 8 sensor features and 3 output targets:

### 1. Input Features (Predictors)
| Feature Name | Type | Recommended Safe Limits (WHO / EPA) | Description |
|---|---|---|---|
| `pH` | Float | 6.5 – 8.5 | Acidity/alkalinity level of water |
| `turbidity_NTU` | Float | < 1.0 NTU (Max 5.0 NTU) | Cloudiness caused by suspended particles |
| `TDS_ppm` | Float | < 300 - 500 ppm | Total Dissolved Solids in parts per million |
| `flow_rate_L_min` | Float | 0.8 – 1.8 L/min | Water flow velocity through purification system |
| `pressure_bar` | Float | 1.5 – 4.5 bar | System operating pressure |
| `temperature_C` | Float | 15.0 – 30.0 °C | Water temperature in Celsius |
| `usage_L_per_day` | Float | 10 – 40 L/day | Daily household/system water consumption |
| `days_since_filter_change` | Integer | < 90 – 120 days | Operational days since last filter replacement |

### 2. Output Targets
| Target Name | Values | Description |
|---|---|---|
| `water_quality` | `0` (Safe), `1` (Moderate), `2` (Unsafe) | Primary classification target for water quality status |
| `filter_replacement` | `0` (No), `1` (Yes) | Auxiliary indicator for filter change requirement |
| `maintenance_required` | `0` (No), `1` (Yes) | Auxiliary indicator for system maintenance alert |

---

## 🤖 Machine Learning Pipeline (`model_training.py`)

1. **Preprocessing & Feature Engineering**:
   - Features (`X`): `pH`, `turbidity_NTU`, `TDS_ppm`, `flow_rate_L_min`, `pressure_bar`, `temperature_C`, `usage_L_per_day`, `days_since_filter_change`
   - Target (`y`): `water_quality`
   - Scaling: `StandardScaler` applied to ensure zero mean and unit variance.

2. **Model Evaluation & Selection**:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier (Optimized)
   - AdaBoost & Gradient Boosting Classifiers

3. **Hyperparameter Optimization & Validation**:
   - `GridSearchCV` cross-validation (5-fold)
   - Performance metrics evaluated: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

4. **Model Export**:
   - Best performing classifier serialized to `model.pkl`.
   - Scaler instance serialized to `scaler.pkl`.

---

## 🖥️ User Interface Applications

### 1. Streamlit Dashboard (`app.py`)
- Sidebar navigation between:
  - **Prediction**: Interactive sliders, safety indicators, parameter threshold evaluation.
  - **Graphs**: Distribution plots, scatter plots, correlation heatmaps.
  - **Model Performance**: Confusion matrix, feature importance rankings, test metrics.

### 2. Gradio Interactive Dashboard (`gradio_app.py`)
- Multi-tab Gradio UI featuring:
  - **Single Prediction**: Live sliders with immediate color-coded result & recommendation cards.
  - **Batch Prediction**: CSV dataset upload & batch inference with downloadable CSV output.
  - **Analytics & Visualizations**: Interactive feature analysis and distribution charts.
  - **Water Safety Benchmarks**: WHO & EPA guideline lookup.

---

## 👩‍💻 Author & Maintainer
- **Janvi Thakre**
- Student, Data Science and Analytics, Ramdeobaba University
