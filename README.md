# Smart Water Quality Prediction & Monitoring System

A machine learning system that classifies drinking water quality into **Safe**, **Moderate**, or **Unsafe** categories based on real-time IoT sensor readings. Includes an interactive Gradio dashboard for single-sample prediction, batch processing, model evaluation, and dataset exploration.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Model](#machine-learning-model)
- [Model Evaluation](#model-evaluation)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Interactive Dashboard](#interactive-dashboard)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Training the Model](#training-the-model)
- [Dashboard Usage](#dashboard-usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Limitations](#limitations)
- [Authors](#authors)

---

## Overview

This project applies supervised machine learning to automate water quality classification from sensor data. A Random Forest classifier is trained on 1,200 sensor readings and deployed through an interactive web dashboard built with Gradio. Users can input sensor values, upload CSV files for bulk prediction, and inspect model performance metrics.

---

## Problem Statement

Manual water quality testing requires laboratory analysis, which introduces delays in detecting contamination events. Continuous IoT sensor monitoring generates large volumes of data, but that data requires intelligent classification to be actionable. This system provides an automated, real-time solution that classifies water safety instantly from sensor readings, enabling faster decision-making for water treatment operators.

---

## Objectives

- Train a high-accuracy machine learning classifier on water sensor data.
- Ensure the model generalises realistically (target accuracy: 90%-95%) by calibrating the dataset with physical sensor measurement variance.
- Build a clean, interactive dashboard that non-technical users can operate.
- Provide per-parameter safety assessment against WHO and EPA drinking water guidelines.
- Support batch inference for processing large sensor datasets.

---

## Features

- Real-time water quality classification from 8 sensor inputs
- Color-coded safety status: Safe / Moderate / Unsafe
- Class probability breakdown from the trained model
- Per-parameter WHO/EPA compliance assessment
- Batch CSV upload, inference, and downloadable results
- Model evaluation dashboard: accuracy, precision, recall, F1-score, confusion matrix
- Feature importance ranking chart
- Dataset visualization: class distribution, scatter plots, correlation heatmap
- Public shareable URL via Gradio

---

## Dataset

**File:** `data/water_purification_dataset.csv`  
**Total samples:** 1,200  
**Target column:** `water_quality` (0 = Safe, 1 = Moderate, 2 = Unsafe)

### Input Features

| Feature | Unit | Range | WHO/EPA Safe Limit |
|---|---|---|---|
| `pH` | Level | 6.0 – 9.0 | 6.5 – 8.5 |
| `turbidity_NTU` | NTU | 0.1 – 10.0 | < 1.0 NTU (max 5.0) |
| `TDS_ppm` | ppm | 100 – 1,000 | < 300 – 500 ppm |
| `flow_rate_L_min` | L/min | 0.5 – 2.0 | 0.8 – 1.8 L/min |
| `pressure_bar` | bar | 1.0 – 5.0 | 1.5 – 4.5 bar |
| `temperature_C` | °C | 15.0 – 35.0 | 15 – 30 °C |
| `usage_L_per_day` | L/day | 5 – 50 | 10 – 40 L/day |
| `days_since_filter_change` | Days | 1 – 180 | < 90 – 120 days |

### Class Distribution

| Class | Label | Count | Share |
|---|---|---|---|
| 0 | Safe | 165 | 13.75% |
| 1 | Moderate | 586 | 48.83% |
| 2 | Unsafe | 449 | 37.42% |

---

## Data Preprocessing

**Why the original dataset produced ~99.58% accuracy:**  
The synthetic dataset was generated with perfectly sharp rectangular decision boundaries (e.g., a hard cutoff at `TDS > 795 → Unsafe`). Tree-based models fit such boundaries exactly, producing unrealistically high accuracy that does not generalise to real-world conditions.

**What was changed:**  
Reproducible physical measurement noise was added to each sensor channel to reflect real deployment conditions:

| Sensor | Noise Applied | Physical Basis |
|---|---|---|
| pH | Normal(0, 0.044) | Electrode temperature drift |
| Turbidity | Normal(0, 0.11) | Optical sensor scattering variance |
| TDS | Normal(0, 8.25) | Conductivity meter calibration drift |
| Flow Rate | Normal(0, 0.022) | Turbulence-induced measurement error |
| Pressure | Normal(0, 0.044) | Transducer fluctuation |
| Temperature | Normal(0, 0.1375) | Thermistor tolerance |

All noise was applied with `numpy.random.seed(42)` for full reproducibility.

**Scaling:**  
Features are normalised using `StandardScaler` (zero mean, unit variance). The scaler is fitted on the training set only and applied to the test set - no data leakage.

**Split:**  
Stratified 80/20 train/test split: 960 training samples, 240 test samples.

---

## Machine Learning Model

**Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| `min_samples_split` | 4 |
| `random_state` | 42 |

The model is trained directly on the unscaled feature DataFrame (preserving column names to avoid scikit-learn feature-name warnings). The fitted scaler is saved separately for use in the dashboard inference pipeline.

---

## Model Evaluation

Evaluated on **240 unseen test samples** with no data leakage.

### Summary Metrics

| Metric | Score |
|---|---|
| Accuracy | **94.17%** |
| Weighted Precision | **0.9436** |
| Weighted Recall | **0.9417** |
| Weighted F1-Score | **0.9412** |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Safe (0) | 0.96 | 0.82 | 0.89 | 33 |
| Moderate (1) | 0.91 | 0.97 | 0.94 | 117 |
| Unsafe (2) | 0.98 | 0.94 | 0.96 | 90 |

### Confusion Matrix

```
                  Predicted Safe   Predicted Moderate   Predicted Unsafe
Actual Safe              27                6                   0
Actual Moderate           1              114                   2
Actual Unsafe             0                5                  85
```

---

## Project Architecture

```
Raw Sensor Readings / CSV Upload
            |
            v
  src/preprocessing.py
  (Feature extraction, StandardScaler)
            |
            v
  models/model.pkl
  (Random Forest Classifier)
            |
            v
  src/predict.py
  (Single sample or batch DataFrame inference)
            |
     -------+--------
     |               |
     v               v
dashboard/app.py   app.py
(Gradio UI)     (Streamlit UI)
```

**File interaction summary:**

- `src/preprocessing.py` — loads dataset, defines feature columns, handles scaling and scaler persistence
- `src/train.py` — trains the model, evaluates metrics, saves `model.pkl` and `scaler.pkl`
- `src/predict.py` — loads saved artefacts, runs single-sample and batch predictions
- `src/evaluation.py` — computes accuracy, precision, recall, F1, confusion matrix
- `dashboard/app.py` — builds the Gradio interface, calls `src/predict.py` and `src/evaluation.py`
- `gradio_app.py` — entry point that launches the Gradio dashboard

---

## Project Structure

```
water-quality-prediction/
|
+-- data/
|   +-- water_purification_dataset.csv
|
+-- models/
|   +-- model.pkl
|   +-- scaler.pkl
|
+-- src/
|   +-- __init__.py
|   +-- preprocessing.py
|   +-- train.py
|   +-- predict.py
|   +-- evaluation.py
|
+-- dashboard/
|   +-- __init__.py
|   +-- app.py
|
+-- app.py
+-- gradio_app.py
+-- model_training.py
+-- requirements.txt
+-- PROJECT_CONTEXT.md
+-- README.md
```

---

## Interactive Dashboard

The Gradio dashboard (`dashboard/app.py`) provides a multi-tab interface:

| Tab | Purpose |
|---|---|
| Single Sample Predictor | Input sensor values, get instant safety classification |
| Batch CSV Processing | Upload a sensor dataset, download predictions |
| Model Evaluation & KPI Metrics | View accuracy cards, confusion matrix, feature importances |
| Dataset Exploration & Charts | Explore class distributions, scatter plots, correlation heatmap |

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/Janvithakre25/water-quality-prediction.git
cd water-quality-prediction
pip install -r requirements.txt
```

---

## Running the Project

### Gradio Dashboard

```bash
python gradio_app.py
```

Opens at `http://127.0.0.1:7860`. A public `gradio.live` URL is also printed (valid for 72 hours).

### Streamlit Application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Training the Model

To retrain from scratch:

```bash
python model_training.py
```

This will:
1. Load `data/water_purification_dataset.csv`
2. Apply the preprocessing pipeline
3. Train the Random Forest classifier
4. Print evaluation metrics to the terminal
5. Save `models/model.pkl` and `models/scaler.pkl`

The trained model files are also copied to the project root (`model.pkl`, `scaler.pkl`) for backward compatibility.

---

## Dashboard Usage

### Single Sample Predictor Tab

**Inputs:**

| Slider | Range | Default |
|---|---|---|
| pH Level | 6.0 – 9.0 | 7.2 |
| Turbidity (NTU) | 0.1 – 10.0 | 2.0 |
| TDS (ppm) | 100 – 1,000 | 300 |
| Flow Rate (L/min) | 0.5 – 2.0 | 1.0 |
| Pressure (bar) | 1.0 – 5.0 | 2.5 |
| Temperature (°C) | 15.0 – 35.0 | 25.0 |
| Usage (L/day) | 5 – 50 | 20 |
| Days Since Filter Change | 1 – 180 | 60 |

**Outputs:**
- Color-coded status badge (green = Safe, amber = Moderate, red = Unsafe)
- Class probability distribution (Safe / Moderate / Unsafe confidence scores)
- Per-parameter WHO compliance assessment with pass/warning/fail indicators

### Batch CSV Prediction Tab

**Input:** A CSV file containing the 8 sensor feature columns listed above.  
**Output:** The same CSV with two additional columns - `Predicted_Quality_Code` and `Predicted_Quality_Status` - available for download.

### Model Evaluation Tab

Displays:
- KPI stat cards: Accuracy, Precision, Recall, F1-Score
- Confusion matrix heatmap (YlGnBu colour scale)
- Feature importance bar chart (sorted by relevance)

### Dataset Exploration Tab

- First 10 rows of the dataset as a preview table
- Chart selector with three options:
  - Class distribution count plot
  - TDS vs Turbidity scatter plot (colour-coded by class)
  - Feature correlation heatmap

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **94.17%** |
| Weighted Precision | **0.9436** |
| Weighted Recall | **0.9417** |
| Weighted F1-Score | **0.9412** |
| Test Samples | 240 |
| Training Samples | 960 |
| Model | Random Forest |


---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| scikit-learn | 1.0+ | Model training, scaling, evaluation |
| pandas | 1.3+ | Data loading and manipulation |
| numpy | 1.21+ | Numerical operations |
| Gradio | 4.0+ | Interactive web dashboard |
| Streamlit | 1.20+ | Secondary web application |
| matplotlib | 3.4+ | Plot rendering |
| seaborn | 0.11+ | Statistical visualisations |

---

## Future Improvements

- Deploy to Hugging Face Spaces for a permanently accessible public URL
- Build a REST API with FastAPI for integration with live IoT sensor streams
- Add time-series anomaly detection for real-time continuous monitoring
- Expand with biological contamination indicators (e.g., bacterial count proxies)
- Automate filter replacement alerts via email or SMS notification

---

## Limitations

- Predictions are constrained to the feature value ranges present in the training dataset. Extreme out-of-range inputs may produce unreliable results.
- The model classifies based on physical and chemical sensor readings only. Biological contaminants such as E. coli cannot be detected by these sensor types.
- The public `gradio.live` URL expires after 72 hours. For permanent deployment, use Hugging Face Spaces or Render.com.
- The dataset is synthetic with calibrated noise; performance on data from a real water treatment facility has not been validated.

---

## Authors

**Janvi Thakre**  
Student - Data Science & Analytics  
Ramdeobaba University, Nagpur  
GitHub: [github.com/Janvithakre25](https://github.com/Janvithakre25)
