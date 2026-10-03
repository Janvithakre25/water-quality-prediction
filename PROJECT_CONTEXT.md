# PROJECT_CONTEXT.md — Smart Water Quality Prediction System

This file is the authoritative technical reference for the project. Read this before making any changes to the codebase, dataset, model, or dashboard.

---

## 1. Project Purpose

The **Smart Water Quality Prediction System** is an end-to-end machine learning application that classifies drinking water quality into three categories — **Safe (0)**, **Moderate (1)**, and **Unsafe (2)** — from 8 real-time IoT sensor readings.

The system comprises:
- A modular Python ML pipeline (`src/`)
- A trained Random Forest classifier (`models/model.pkl`)
- An interactive Gradio dashboard with single prediction, batch CSV processing, model metrics, and dataset exploration (`dashboard/app.py`)
- A secondary Streamlit web application (`app.py`)

---

## 2. Directory Structure

```
water-quality-prediction/
|
+-- data/
|   +-- water_purification_dataset.csv   # 1,200 sensor readings (calibrated with noise)
|
+-- models/
|   +-- model.pkl                        # Trained RandomForestClassifier (saved via pickle)
|   +-- scaler.pkl                       # Fitted StandardScaler (saved via pickle)
|
+-- src/
|   +-- __init__.py
|   +-- preprocessing.py                 # FEATURE_COLS, load_data, split_and_scale_data, save/load scaler
|   +-- train.py                         # train_and_save_model() — full training + evaluation pipeline
|   +-- predict.py                       # predict_single_sample(), predict_batch_df(), load_trained_model()
|   +-- evaluation.py                    # evaluate_model_performance() — accuracy, precision, recall, F1, CM
|
+-- dashboard/
|   +-- __init__.py
|   +-- app.py                           # Full Gradio dashboard (4 tabs, HTML badges, KPI cards, charts)
|
+-- app.py                               # Streamlit application (navigation: prediction / analytics / metrics)
+-- gradio_app.py                        # Gradio launcher (entry point — calls dashboard/app.py)
+-- model_training.py                    # Training launcher (entry point — calls src/train.py)
+-- water_purification_dataset.csv       # Root copy of dataset (kept for backward compatibility)
+-- requirements.txt                     # Python package dependencies
+-- .gitignore                           # Excludes: __pycache__, .gradio/, batch output CSV, logs
+-- PROJECT_CONTEXT.md                   # This file
+-- CONTEXT.md                           # Short context reference (points to this file)
+-- README.md                            # Full project documentation
```

---

## 3. Dataset

**File:** `data/water_purification_dataset.csv`  
**Samples:** 1,200  
**Columns:** 11 total (8 features + 3 targets)

### Input Feature Columns (used as model inputs)

| Column | Type | Range | Description |
|---|---|---|---|
| `pH` | float | 6.0 – 9.0 | Water acidity / alkalinity |
| `turbidity_NTU` | float | 0.1 – 10.0 | Suspended particles / optical clarity |
| `TDS_ppm` | float | 100 – 1,000 | Total dissolved solids |
| `flow_rate_L_min` | float | 0.5 – 2.0 | Filtration system flow velocity |
| `pressure_bar` | float | 1.0 – 5.0 | System operating pressure |
| `temperature_C` | float | 15.0 – 35.0 | Water temperature |
| `usage_L_per_day` | float | 5 – 50 | Daily consumption volume |
| `days_since_filter_change` | int | 1 – 180 | Active days since last filter service |

### Output / Target Columns

| Column | Values | Used As |
|---|---|---|
| `water_quality` | 0 (Safe), 1 (Moderate), 2 (Unsafe) | **Primary prediction target** |
| `filter_replacement` | 0, 1 | Auxiliary — not used in model training |
| `maintenance_required` | 0, 1 | Auxiliary — not used in model training |

### Class Distribution

| Class | Label | Count | Share |
|---|---|---|---|
| 0 | Safe | 165 | 13.75% |
| 1 | Moderate | 586 | 48.83% |
| 2 | Unsafe | 449 | 37.42% |

---

## 4. Why Accuracy Was ~99.58% Originally

The original synthetic dataset used perfectly sharp rectangular decision boundaries (e.g., `TDS > 795 → Unsafe`, `turbidity > 7.99 → Unsafe`). Tree-based classifiers fit such boundaries exactly, producing unrealistically high accuracy.

This is a **synthetic data overfitting problem**, not a model bug.

---

## 5. Dataset Calibration Fix

Reproducible physical sensor measurement noise was applied (`numpy.random.seed(42)`) to blur the artificial boundaries and simulate real-world sensor deployments:

| Sensor | Std Dev Applied | Physical Basis |
|---|---|---|
| pH | 0.044 | Electrode temperature drift |
| turbidity_NTU | 0.11 | Optical scattering variance |
| TDS_ppm | 8.25 | Conductivity meter calibration drift |
| flow_rate_L_min | 0.022 | Turbulence-induced measurement error |
| pressure_bar | 0.044 | Transducer pressure fluctuation |
| temperature_C | 0.1375 | Thermistor tolerance |

The calibrated dataset is saved to both `data/water_purification_dataset.csv` and the project root `water_purification_dataset.csv`.

Result: Accuracy dropped from ~99.58% to **94.17%**.

---

## 6. Preprocessing Pipeline

Defined in `src/preprocessing.py`.

**`FEATURE_COLS`** (global constant used across all modules):
```python
["pH", "turbidity_NTU", "TDS_ppm", "flow_rate_L_min",
 "pressure_bar", "temperature_C", "usage_L_per_day", "days_since_filter_change"]
```

**Steps:**
1. `load_data()` — tries `data/`, then root directory
2. `prepare_features_and_target()` — extracts `X` (FEATURE_COLS) and `y` (water_quality)
3. `split_and_scale_data()` — stratified 80/20 split, fits `StandardScaler` on train set only
4. `save_scaler()` / `load_scaler()` — saves to `models/scaler.pkl` and root `scaler.pkl`

**Important:** The scaler is fitted on training data only. Test data is transformed using the already-fitted scaler. No data leakage.

---

## 7. Model Training Pipeline

Defined in `src/train.py`. Entry point: `model_training.py`.

```
python model_training.py
```

**Algorithm:** `RandomForestClassifier`

| Hyperparameter | Value | Reason |
|---|---|---|
| `n_estimators` | 100 | Stable ensemble without excessive computation |
| `max_depth` | 10 | Limits overfitting on noisy calibrated data |
| `min_samples_split` | 4 | Prevents overly fine splits on small leaf nodes |
| `random_state` | 42 | Full reproducibility |

**Training flow:**
1. Load data → extract features → split → scale → fit scaler
2. Fit `RandomForestClassifier` on `X_train` (raw DataFrame, not scaled — RF does not require scaling)
3. Evaluate on `X_test` using `evaluate_model_performance()`
4. Save model to `models/model.pkl` and root `model.pkl`
5. Save scaler to `models/scaler.pkl` and root `scaler.pkl`

**Note:** The model is trained on the raw (unscaled) DataFrame to preserve sklearn feature names and avoid `UserWarning` during inference.

---

## 8. Prediction Pipeline

Defined in `src/predict.py`.

**`load_trained_model()`** — searches `models/model.pkl`, then root `model.pkl`

**`predict_single_sample(input_values, model, scaler)`**
- Accepts: dict (keyed by FEATURE_COLS), list, ndarray, or DataFrame
- Wraps input in a pandas DataFrame with named columns before calling `model.predict()` — prevents sklearn UserWarning
- Returns: `(pred_code, status_label, probabilities_dict)`

**`predict_batch_df(df, model, scaler)`**
- Accepts: DataFrame with required sensor columns
- Returns: original DataFrame + two new columns (`Predicted_Quality_Code`, `Predicted_Quality_Status`)

**Label map:**
```python
{0: "Safe ✅", 1: "Moderate ⚠️", 2: "Unsafe ❌"}
```

---

## 9. Evaluation Module

Defined in `src/evaluation.py`.

**`evaluate_model_performance(model, X_test, y_test, is_scaled=False, scaler=None)`**

Returns a dict with:
- `accuracy` — overall fraction correct
- `precision` — weighted average precision
- `recall` — weighted average recall
- `f1_score` — weighted average F1
- `confusion_matrix` — numpy array (3×3)
- `classification_report` — formatted string
- `predictions` — array of predicted labels

---

## 10. Final Model Evaluation Results

Evaluated on **240 unseen test samples** (stratified 20% split). No data leakage.

| Metric | Value |
|---|---|
| Accuracy | **94.17%** (226 / 240 correct) |
| Weighted Precision | **0.9436** |
| Weighted Recall | **0.9417** |
| Weighted F1-Score | **0.9412** |

**Per-class breakdown:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Safe (0) | 0.96 | 0.82 | 0.89 | 33 |
| Moderate (1) | 0.91 | 0.97 | 0.94 | 117 |
| Unsafe (2) | 0.98 | 0.94 | 0.96 | 90 |

**Confusion Matrix:**
```
              Predicted Safe   Predicted Moderate   Predicted Unsafe
Actual Safe        27                 6                   0
Actual Moderate     1               114                   2
Actual Unsafe       0                 5                  85
```

---

## 11. Gradio Dashboard

Defined in `dashboard/app.py`. Entry point: `gradio_app.py`.

```
python gradio_app.py
```

**Launch settings:**
- Local: `http://127.0.0.1:7860`
- Public: `share=True` generates a temporary `gradio.live` URL (72-hour expiry)
- Theme: moved from `gr.Blocks(theme=...)` to `app.launch(theme=...)` to comply with Gradio 6.0 API

**Tab 1 — Single Sample Predictor:**
- 8 sensor sliders grouped into Physical Properties and Operational Parameters sections
- Outputs: HTML color-coded status badge, `gr.Label` class probability bar, HTML per-parameter WHO assessment
- Reset button restores all sliders to default values

**Tab 2 — Batch CSV Processing:**
- CSV file upload (`gr.File`)
- Validates required columns match `FEATURE_COLS`
- Output: `gr.Dataframe` preview + `gr.File` download of processed CSV

**Tab 3 — Model Evaluation & KPI Metrics:**
- 4 HTML KPI stat cards: Accuracy / Precision / Recall / F1
- Confusion matrix heatmap (YlGnBu, 100 dpi, no colorbar)
- Feature importance bar chart (sorted ascending, crest palette, 100 dpi)

**Tab 4 — Dataset Exploration:**
- First 10 rows preview table
- Dropdown to select: Class Distribution / TDS vs Turbidity Scatter / Correlation Heatmap
- All charts 7.5×4.5 inches, 100 dpi

---

## 12. Streamlit Application

Defined in `app.py`. Entry point: `streamlit run app.py`.

Sidebar navigation with three views:
- **Single Prediction** — same 8 sliders, success/warning/error status boxes, per-parameter evaluation
- **Dataset Analytics** — class distribution countplot
- **Model Metrics** — accuracy/precision/recall/F1 markdown + confusion matrix heatmap

Uses `@st.cache_resource` for model loading and `@st.cache_data` for dataset loading.

---

## 13. File Interaction Map

```
model_training.py
    └── src/train.py
            ├── src/preprocessing.py  (load_data, split_and_scale_data, save_scaler)
            └── src/evaluation.py     (evaluate_model_performance)
                    └── saves: models/model.pkl, models/scaler.pkl

gradio_app.py
    └── dashboard/app.py
            ├── src/preprocessing.py  (FEATURE_COLS, load_data, load_scaler)
            ├── src/predict.py        (load_trained_model, predict_single_sample, predict_batch_df)
            └── src/evaluation.py     (evaluate_model_performance)

app.py (Streamlit)
    ├── src/preprocessing.py
    ├── src/predict.py
    └── src/evaluation.py
```

---

## 14. Execution Reference

| Task | Command |
|---|---|
| Train / retrain model | `python model_training.py` |
| Launch Gradio dashboard | `python gradio_app.py` |
| Launch Streamlit app | `streamlit run app.py` |
| Run via module | `python -m src.train` |
| Install dependencies | `pip install -r requirements.txt` |

---

## 15. Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
gradio>=4.0.0
streamlit>=1.20.0
```

---

## 16. Known Issues & Fixes Applied

| Issue | Cause | Fix Applied |
|---|---|---|
| `UserWarning: X does not have valid feature names` | Passing raw arrays to sklearn model trained on DataFrame | `predict_single_sample()` now wraps input in a named-column DataFrame before calling `model.predict()` |
| `ModuleNotFoundError: No module named 'gradio'` | Gradio not installed in environment | Run `pip install -r requirements.txt` |
| Gradio 6.0 `theme` parameter warning | `theme` passed to `gr.Blocks()` instead of `launch()` | Moved to `app.launch(theme=...)` in `dashboard/app.py` |
| ~99.58% unrealistic accuracy | Synthetic sharp boundaries in dataset | Calibrated sensor noise applied (seed=42) |

---

## 17. Rules for Future Modifications

1. Never manipulate evaluation metrics post-calculation. All reported metrics must be empirically computed.
2. Always use `FEATURE_COLS` from `src/preprocessing.py` as the single source of truth for column ordering.
3. Keep the scaler fitted on training data only. Never fit on the full dataset.
4. Preserve `random_state=42` and `test_size=0.2` in `split_and_scale_data()` to maintain reproducibility.
5. Pass DataFrames with named columns to `model.predict()` to avoid sklearn feature-name warnings.
6. Do not add `.gradio/`, `__pycache__/`, `batch_predictions_output.csv`, or `.env` files to git.
7. Update this file whenever the dataset, model, preprocessing pipeline, or dashboard changes significantly.
