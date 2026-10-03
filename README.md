# 💧 Smart Water Purification System & Quality Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/Gradio-Dashboard-FF5500.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project that classifies water quality (**Safe ✅**, **Moderate ⚠️**, **Unsafe ❌**) based on IoT sensor data and operational parameters. Features both a **Streamlit Web App** and an interactive **Gradio Dashboard** with single & batch CSV inference capabilities.

---

## 📌 Features

- 🔮 **Water Quality Classification**: Predicts water safety using 8 real-time sensor parameters.
- ⚡ **Interactive Gradio Dashboard (`gradio_app.py`)**: Multi-tab interface featuring real-time input sliders, probability visualizer, and WHO safety parameter benchmarks.
- 📁 **Batch Inference**: Upload CSV files containing sensor readings and download predictions with safety labels.
- 📊 **Streamlit Web Application (`app.py`)**: Includes interactive slider predictions, dataset exploratory data analysis (EDA), and confusion matrix metrics.
- 🤖 **Automated ML Training & Tuning (`model_training.py`)**: Evaluates multiple classifiers (Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting) with 5-fold cross-validation and hyperparameter optimization via `GridSearchCV`.

---

## 📁 Repository Structure

```
water-quality-prediction/
├── app.py                          # Streamlit Web Application
├── gradio_app.py                   # Interactive Gradio Dashboard (Single & Batch Prediction)
├── model_training.py               # Machine learning training, tuning, and evaluation script
├── water_purification_dataset.csv  # Sensor readings dataset
├── model.pkl                       # Trained machine learning model
├── scaler.pkl                      # Fitted StandardScaler instance
├── requirements.txt                # Python package dependencies
├── PROJECT_CONTEXT.md              # Technical project context & dataset specifications
├── CONTEXT.md                      # Context reference file
└── README.md                       # Documentation & guide
```

---

## 📊 Dataset & Sensor Features

The model consumes 8 sensor parameters to predict water quality:

| Parameter | Unit | WHO / EPA Recommended Range | Description |
|---|---|---|---|
| **pH** | Level | 6.5 – 8.5 | Acidity / Alkalinity level |
| **Turbidity** | NTU | < 1.0 (Max 5.0) | Water cloudiness & suspended particles |
| **TDS** | ppm | < 300 – 500 ppm | Total Dissolved Solids |
| **Flow Rate** | L/min | 0.8 – 1.8 L/min | System filtration flow speed |
| **Pressure** | bar | 1.5 – 4.5 bar | System operating pressure |
| **Temperature** | °C | 15.0 – 30.0 °C | Water temperature |
| **Daily Usage** | L/day | 10 – 40 L/day | Daily consumption volume |
| **Days Since Filter Change** | Days | < 90 – 120 Days | Operational filter usage days |

---

## 🤖 Machine Learning Workflow & Performance

Models evaluated during training:
- **Decision Tree Classifier** (~99.58% accuracy)
- **Gradient Boosting Classifier** (~99.58% accuracy)
- **Random Forest Classifier** (~98.75% accuracy)
- **Logistic Regression** (~72.08% accuracy)
- **AdaBoost Classifier** (~68.33% accuracy)

The optimal model is hyperparameter-tuned, validated via 5-fold cross-validation, and serialized as `model.pkl` along with `scaler.pkl`.

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Janvithakre25/water-quality-prediction.git
cd water-quality-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train / Re-train the Machine Learning Model
```bash
python model_training.py
```

---

## 🖥️ Running the User Interfaces

### Option A: Launch Gradio Interactive Dashboard
```bash
python gradio_app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your web browser.

### Option B: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your web browser.

---

## 🙋 Author & Contact

**Janvi Thakre**  
Student — Data Science & Analytics, Ramdeobaba University  
GitHub: [@Janvithakre25](https://github.com/Janvithakre25)
