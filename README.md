# 💧 Smart Water Purification System

Hi! This is my mini project where I built a machine learning model that predicts whether water is **Safe, Moderate, or Unsafe** based on sensor readings. I also made a simple web app using Streamlit so anyone can test it without knowing code.

---

## 📌 What is this project about?

Water quality is a serious issue and I wanted to apply what I learned in ML to a real-world problem. So I trained a classification model on a water purification dataset and built an interactive app around it.

The app lets you:
- Enter sensor values (like pH, TDS, turbidity etc.) and predict water quality
- See some graphs from the dataset
- Compare your water readings against safe limits

---

## 📁 Files in this project

```
├── app.py                          # Streamlit app (the UI part)
├── model_training.py               # All the ML code (EDA, training, saving model)
├── water_purification_dataset.csv  # Dataset I used
├── model.pkl                       # Saved trained model
├── scaler.pkl                      # Saved scaler (needed for preprocessing)
└── README.md                       # This file
```

> Note: `model.pkl` and `scaler.pkl` are generated when you run `model_training.py`

---

## 🤖 Machine Learning Part

I tried 4 different models and compared their accuracy:

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost

The best model is automatically selected and then:
1. Cross validated (5-fold)
2. Hyperparameter tuned using GridSearchCV (for Decision Tree and Random Forest)
3. Saved as `model.pkl`

I also saved the `StandardScaler` as `scaler.pkl` because the same scaling needs to be applied when predicting new inputs.

---

## 🖥️ About the App

The Streamlit app has 3 sections:

### 🔍 Prediction
You enter 8 values using sliders:

| Input | Range |
|---|---|
| pH | 6.0 – 9.0 |
| Turbidity (NTU) | 0.1 – 10.0 |
| TDS (ppm) | 100 – 1000 |
| Flow Rate (L/min) | 0.5 – 2.0 |
| Pressure (bar) | 1.0 – 5.0 |
| Temperature (°C) | 15.0 – 35.0 |
| Usage (L/day) | 5 – 50 |
| Days Since Filter Change | 1 – 180 |

And it tells you if the water is:
- ✅ SAFE
- ⚠️ MODERATE
- ❌ UNSAFE

### 📊 Graphs
Some basic visualizations from the dataset like count plots and scatter plots.

### 📈 User Analysis
You can enter pH, Turbidity and TDS values and it shows a bar chart comparing your values to safe limits, plus individual status messages for each.

---

## ▶️ How to run this

### Step 1 – Clone the repo
```bash
git clone https://github.com/your-username/smart-water-purification.git
cd smart-water-purification
```

### Step 2 – Install required libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit
```

### Step 3 – Train the model first
```bash
python model_training.py
```
This will create `model.pkl` and `scaler.pkl` in the same folder.

### Step 4 – Run the app
```bash
streamlit run app.py
```

---

## 📦 Libraries Used

- `numpy`, `pandas` – data handling
- `scikit-learn` – ML models, scaling, evaluation
- `matplotlib`, `seaborn` – visualizations
- `streamlit` – web app
- `pickle` – saving and loading the model

---

## 📋 Dataset

The dataset has sensor readings from a water purification system. Target column is `water_quality` (Safe / Moderate / Unsafe). I dropped `filter_replacement` and `maintenance_required` columns since they are not sensor inputs.

---

## 📝 Notes

- This project was made as part of my learning in Machine Learning.
- The model might not be perfect but it works well on the test data.
- Feel free to use or modify this project.

---

## 🙋 Made by

**[Your Name]**  
Student – [Your Course / College Name]
