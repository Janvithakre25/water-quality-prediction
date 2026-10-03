# Smart Water Quality Prediction & Monitoring System

## Overview
The Smart Water Quality Prediction System is an end-to-end machine learning platform that evaluates water safety based on real-time IoT sensor readings. It classifies water samples into three distinct quality levels: **Safe (0)**, **Moderate (1)**, and **Unsafe (2)**. The system includes a modular Python backend, model training pipelines, a Streamlit web application, and an interactive Gradio dashboard supporting both real-time single-sample inference and batch CSV processing.

## Problem Statement
Access to clean drinking water is vital for public health. Manual laboratory testing of water quality can be time-consuming and expensive. Utilizing machine learning models trained on sensor readings (such as pH, turbidity, total dissolved solids, and system pressure) enables automated, immediate quality classification and proactive filtration system maintenance.

## Objectives
- Develop a machine learning model capable of accurately predicting water quality from sensor readings.
- Calibrate model training on realistic sensor variance to avoid overfitting and ensure realistic generalization performance (target accuracy between 90% and 95%).
- Build an interactive Gradio web application for real-time predictions, batch CSV processing, and dataset exploration.
- Establish a clean, modular repository structure (`src/`, `data/`, `models/`, `dashboard/`) for maintainability and reproducibility.

## Features
- **Real-Time Quality Inference**: Predict water quality instantly using interactive sliders for sensor inputs.
- **WHO & EPA Parameter Guidance**: Receive automated assessment notes based on standard WHO safe ranges.
- **Batch CSV Processing**: Upload CSV files with sensor data, execute batch predictions, preview output tables, and download processed CSV files.
- **Model Evaluation Dashboard**: View live model evaluation metrics, feature importances, and confusion matrix visualizer.
- **Dataset Analytics**: Explore class distributions, feature scatter plots, and correlation heatmaps.

## Dataset
The dataset (`data/water_purification_dataset.csv`) consists of 1,200 samples containing 8 input features and 1 categorical target variable:

| Feature Column | Unit | Range | Description |
|---|---|---|---|
| `pH` | Level | 6.0 – 9.0 | Acidity / Alkalinity measure |
| `turbidity_NTU` | NTU | 0.1 – 10.0 | Water clarity & suspended solids |
| `TDS_ppm` | ppm | 100 – 1000 | Total Dissolved Solids |
| `flow_rate_L_min` | L/min | 0.5 – 2.0 | Filtration flow rate |
| `pressure_bar` | bar | 1.0 – 5.0 | System operating pressure |
| `temperature_C` | °C | 15.0 – 35.0 | Water temperature |
| `usage_L_per_day` | L/day | 5 – 50 | Daily consumption volume |
| `days_since_filter_change` | Days | 1 – 180 | Active days since filter change |
| `water_quality` | Class | 0, 1, 2 | Target: Safe (0), Moderate (1), Unsafe (2) |

## Data Preprocessing
- **Sensor Calibration Variance**: Refined synthetic sharp step boundaries with realistic physical measurement variance (e.g., optical turbidity sensor noise, pH temperature sensitivity, and electrical conductivity drift) to reflect real-world sensor deployments.
- **Feature Scaling**: Input variables are normalized using `StandardScaler` to ensure zero mean and unit variance.
- **Train/Test Splitting**: Stratified 80/20 train/test split (960 training samples, 240 evaluation samples).

## Machine Learning Model
The system uses a **Random Forest Classifier** configured with 100 estimators, a maximum tree depth of 10, and a minimum sample split of 4.

## Model Evaluation
The model was evaluated on an independent 20% test split. The empirical performance metrics are:

- **Accuracy**: **94.17%** (226 / 240 samples correctly classified)
- **Precision**: **0.9436**
- **Recall**: **0.9417**
- **F1-Score**: **0.9412**

### Confusion Matrix
```text
               Predicted Safe (0)   Predicted Moderate (1)   Predicted Unsafe (2)
Actual Safe (0)        27                      6                      0
Actual Moderate (1)     1                    114                      2
Actual Unsafe (2)       0                      5                     85
```

## Project Architecture
```text
Sensor Inputs / CSV Upload
          │
          ▼
┌───────────────────────────┐
│   src/preprocessing.py    │ ── (StandardScaler Normalization)
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│     models/model.pkl      │ ── (Random Forest Classifier)
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      src/predict.py       │ ── (Output: Quality Code & Probabilities)
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│     dashboard/app.py      │ ── (Gradio Interactive Interface)
└───────────────────────────┘
```

## Project Structure
```text
water-quality-prediction/
├── data/
│   └── water_purification_dataset.csv
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluation.py
├── dashboard/
│   ├── __init__.py
│   └── app.py
├── app.py
├── gradio_app.py
├── model_training.py
├── PROJECT_CONTEXT.md
├── requirements.txt
└── README.md
```

## Interactive Dashboard
The system includes an interactive **Gradio Dashboard** built with custom soft themes and modular tabs:
1. **Single Prediction Tab**: Input sliders, status badges, probability breakdowns, and WHO guideline compliance advice.
2. **Batch CSV Prediction Tab**: File uploader, batch execution engine, output data preview, and CSV downloader.
3. **Model Performance Tab**: Actual accuracy metrics, feature importances, and interactive confusion matrix plots.
4. **Dataset Exploration Tab**: Dataset table preview and interactive exploratory charts.

## Installation

```bash
# Clone the repository
git clone https://github.com/Janvithakre25/water-quality-prediction.git
cd water-quality-prediction

# Install required dependencies
pip install -r requirements.txt
```

## Running the Project

### Launch Gradio Dashboard (Recommended)
```bash
python gradio_app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your web browser.

### Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your web browser.

## Training the Model
To re-train and evaluate the machine learning model:
```bash
python model_training.py
# or
python -m src.train
```

## Dashboard Usage
1. Adjust the sensor input sliders (pH, Turbidity, TDS, Flow Rate, Pressure, Temp, Usage, Days).
2. Click **⚡ Predict Quality** to view the safety classification and WHO compliance assessment.
3. Click **🔄 Reset Inputs** to restore default sensor values.
4. Switch to **Batch CSV Prediction** tab to upload a custom CSV file and download predicted results.

## Results
- Evaluated Accuracy: **94.17%**
- Evaluated Weighted F1-Score: **0.9412**
- Validated on 240 unseen test samples with zero data leakage.

## Technologies Used
- **Python 3.9+**
- **Scikit-Learn**: Machine learning training, scaling, evaluation
- **Pandas & NumPy**: Data processing and matrix operations
- **Gradio**: Interactive web application framework
- **Streamlit**: Secondary web application framework
- **Matplotlib & Seaborn**: Data visualizations and confusion matrix plots

## Future Improvements
- Integrate time-series anomaly detection for real-time IoT sensor streams.
- Deploy containerized application via Docker to cloud services (Hugging Face Spaces / AWS).

## Limitations
- Model predictions rely on the range of sensor metrics present in the dataset (e.g., pH 6.0 – 9.0).
- Extreme biological contaminants requiring microbial analysis are not measured directly by basic physical sensors.

## Authors
**Janvi Thakre**  
Student — Data Science & Analytics, Ramdeobaba University  
GitHub: [@Janvithakre25](https://github.com/Janvithakre25)
