
# Smart Water Quality Prediction & Monitoring System

## Overview
An end-to-end Machine Learning dashboard that classifies water quality into **Safe**, **Moderate**, or **Unsafe** categories based on 8 real-time IoT sensor readings. Built with a Random Forest Classifier achieving **94.17% test accuracy**.

## Features
- **Real-time Single Prediction** — Adjust 8 sensor sliders and instantly get safety classification with WHO compliance assessment
- **Batch CSV Processing** — Upload datasets and download predictions in bulk
- **Model Metrics Dashboard** — Confusion matrix, feature importances, and KPI cards
- **Dataset Analytics** — Interactive scatter plots, distributions, and correlation heatmaps
- **WHO/EPA Standards Reference** — Built-in safe limit lookup table

## Sensor Parameters Used
| Parameter | Safe Range |
|---|---|
| pH | 6.5 – 8.5 |
| Turbidity | < 5.0 NTU |
| TDS | < 500 ppm |
| Flow Rate | 0.8 – 1.8 L/min |
| Pressure | 1.5 – 4.5 bar |
| Temperature | 15 – 30 °C |
| Daily Usage | 10 – 40 L/day |
| Days Since Filter Change | < 90 days |

## Model Performance
- **Accuracy**: 94.17%
- **Precision**: 0.9436
- **Recall**: 0.9417
- **F1-Score**: 0.9412

## Running Locally
```bash
pip install -r requirements.txt
python model_training.py   # train the model
python gradio_app.py       # launch with public share link
```

## Author
Janvi Thakre — Data Science & Analytics, Ramdeobaba University  
GitHub: [@Janvithakre25](https://github.com/Janvithakre25)
