import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# HELPER FUNCTIONS & ARTIFACTS
# -----------------------------
def load_model_and_scaler():
    if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
        return None, None
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

def load_dataset():
    if os.path.exists("water_purification_dataset.csv"):
        return pd.read_csv("water_purification_dataset.csv")
    return None

df_data = load_dataset()

# Status Mappings
QUALITY_MAP = {
    0: ("SAFE ✅", "green", "Water parameters are within optimal safety ranges. Safe for consumption."),
    1: ("MODERATE ⚠️", "orange", "Water parameters show moderate deviation. Pre-treatment or filter check advised."),
    2: ("UNSAFE ❌", "red", "Water parameters exceed safe safety standards! Immediate servicing required.")
}

# -----------------------------
# GRADIO TAB FUNCTIONS
# -----------------------------
def predict_single(pH, turbidity, tds, flow, pressure, temp, usage, days):
    if model is None or scaler is None:
        return "<h3 style='color:red;'>Model or Scaler not loaded! Run model_training.py first.</h3>", {}, ""
        
    input_data = np.array([[pH, turbidity, tds, flow, pressure, temp, usage, days]])
    input_scaled = scaler.transform(input_data)
    
    pred = model.predict(input_scaled)[0]
    label_text, color, desc = QUALITY_MAP[pred]
    
    # Class Probabilities if supported
    probs = {}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        probs = {
            "Safe (0)": float(probabilities[0]),
            "Moderate (1)": float(probabilities[1]),
            "Unsafe (2)": float(probabilities[2])
        }
    else:
        probs = {label_text: 1.0}
        
    # Parameter threshold checks
    notes = []
    if not (6.5 <= pH <= 8.5):
        notes.append(f"⚠️ **pH ({pH})**: Outside WHO optimal range (6.5 - 8.5).")
    else:
        notes.append(f"✅ **pH ({pH})**: Optimal.")
        
    if turbidity > 5.0:
        notes.append(f"⚠️ **Turbidity ({turbidity} NTU)**: High turbidity detected (> 5.0 NTU).")
    else:
        notes.append(f"✅ **Turbidity ({turbidity} NTU)**: Normal.")
        
    if tds > 500:
        notes.append(f"⚠️ **TDS ({tds} ppm)**: High total dissolved solids (> 500 ppm).")
    else:
        notes.append(f"✅ **TDS ({tds} ppm)**: Acceptable.")
        
    if days > 90:
        notes.append(f"🔧 **Filter Age ({days} days)**: Filter replacement check recommended (> 90 days).")
        
    assessment_html = f"""
    <div style='background-color: rgba(240, 240, 240, 0.5); padding: 15px; border-radius: 8px; border-left: 6px solid {color};'>
        <h2 style='color: {color}; margin-top: 0;'>Water Quality: {label_text}</h2>
        <p><b>Assessment Summary:</b> {desc}</p>
    </div>
    """
    
    notes_markdown = "### 📋 Rule-Based Health Checklist:\n" + "\n".join(notes)
    
    return assessment_html, probs, notes_markdown

def batch_predict(file_obj):
    if model is None or scaler is None:
        return None, "Model/Scaler missing.", None
    if file_obj is None:
        return None, "Please upload a CSV file.", None
        
    try:
        df_upload = pd.read_csv(file_obj.name)
        feature_cols = ["pH", "turbidity_NTU", "TDS_ppm", "flow_rate_L_min",
                        "pressure_bar", "temperature_C", "usage_L_per_day", "days_since_filter_change"]
                        
        missing_cols = [c for c in feature_cols if c not in df_upload.columns]
        if missing_cols:
            return None, f"Error: CSV missing required columns: {missing_cols}", None
            
        X = df_upload[feature_cols]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        df_upload["Predicted_Quality_Code"] = predictions
        df_upload["Predicted_Quality_Label"] = df_upload["Predicted_Quality_Code"].map({0: "SAFE", 1: "MODERATE", 2: "UNSAFE"})
        
        # Output plot
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(x="Predicted_Quality_Label", data=df_upload, palette="viridis", ax=ax)
        ax.set_title("Batch Prediction Results Summary")
        
        # Save temp CSV output
        out_path = "batch_predictions_output.csv"
        df_upload.to_csv(out_path, index=False)
        
        return df_upload, f"Successfully processed {len(df_upload)} samples!", fig
    except Exception as e:
        return None, f"Failed to process CSV: {str(e)}", None

def get_eda_plots():
    if df_data is None:
        return None, None
        
    # Plot 1: Class distribution
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x="water_quality", data=df_data, palette="Set2", ax=ax1)
    ax1.set_xticklabels(["Safe (0)", "Moderate (1)", "Unsafe (2)"])
    ax1.set_title("Target Class Distribution")
    
    # Plot 2: Heatmap correlation
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    ax2.set_title("Feature Correlation Heatmap")
    
    return fig1, fig2

# -----------------------------
# BUILD GRADIO DASHBOARD
# -----------------------------
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan"
)

with gr.Blocks(theme=theme, title="💧 Smart Water Quality Dashboard") as demo:
    gr.Markdown("""
    # 💧 Smart Water Purification System - Interactive Dashboard
    ### AI-Powered Multi-Sensor Telemetry & Water Safety Classifier
    """)
    
    with gr.Tabs():
        # TAB 1: Single Prediction
        with gr.TabItem("🔍 Single Sample Prediction"):
            gr.Markdown("#### Adjust telemetry parameters to get real-time water quality predictions.")
            with gr.Row():
                with gr.Column():
                    pH = gr.Slider(6.0, 9.0, value=7.2, step=0.1, label="pH Level")
                    turbidity = gr.Slider(0.1, 10.0, value=2.5, step=0.1, label="Turbidity (NTU)")
                    tds = gr.Slider(100, 1000, value=350, step=10, label="TDS (ppm)")
                    flow = gr.Slider(0.5, 2.0, value=1.2, step=0.05, label="Flow Rate (L/min)")
                    pressure = gr.Slider(1.0, 5.0, value=2.8, step=0.1, label="Pressure (bar)")
                    temp = gr.Slider(15.0, 35.0, value=24.0, step=0.5, label="Temperature (°C)")
                    usage = gr.Slider(5, 50, value=25, step=1, label="Daily Usage (L/day)")
                    days = gr.Slider(1, 180, value=45, step=1, label="Days Since Filter Change")
                    
                    predict_btn = gr.Button("🚀 Classify Water Quality", variant="primary")
                    
                with gr.Column():
                    result_html = gr.HTML(label="Result")
                    probs_label = gr.Label(label="Class Probability Breakdown")
                    checklist_md = gr.Markdown()
                    
            predict_btn.click(
                fn=predict_single,
                inputs=[pH, turbidity, tds, flow, pressure, temp, usage, days],
                outputs=[result_html, probs_label, checklist_md]
            )

        # TAB 2: Batch CSV Upload
        with gr.TabItem("📂 Batch CSV Prediction"):
            gr.Markdown("#### Upload a CSV file containing telemetry readings to generate predictions for all rows.")
            file_input = gr.File(label="Upload Water Telemetry CSV", file_types=[".csv"])
            batch_btn = gr.Button("⚡ Process Batch Predictions", variant="primary")
            
            status_text = gr.Textbox(label="Batch Process Status")
            with gr.Row():
                output_table = gr.DataFrame(label="Prediction Results Preview")
                output_plot = gr.Plot(label="Batch Quality Distribution")
                
            batch_btn.click(
                fn=batch_predict,
                inputs=[file_input],
                outputs=[output_table, status_text, output_plot]
            )

        # TAB 3: Exploratory Data Analysis
        with gr.TabItem("📊 Dataset Exploratory Analysis"):
            gr.Markdown("#### View exploratory data distribution graphs and feature correlations.")
            eda_btn = gr.Button("🔄 Load / Refresh EDA Plots")
            with gr.Row():
                plot1 = gr.Plot(label="Target Class Breakdown")
                plot2 = gr.Plot(label="Feature Correlation Matrix")
                
            eda_btn.click(fn=get_eda_plots, inputs=[], outputs=[plot1, plot2])

        # TAB 4: Model Info & Overview
        with gr.TabItem("ℹ️ Model Architecture & Info"):
            gr.Markdown("""
            ### 🤖 Machine Learning Model Architecture
            - **Algorithm**: Decision Tree Classifier (Optimized via GridSearchCV)
            - **Scaler**: `StandardScaler` (Z-score normalization)
            - **Features Used (8)**: `pH`, `turbidity_NTU`, `TDS_ppm`, `flow_rate_L_min`, `pressure_bar`, `temperature_C`, `usage_L_per_day`, `days_since_filter_change`
            - **Target Classes**:
              - `0`: SAFE ✅ (Optimal drinking water conditions)
              - `1`: MODERATE ⚠️ (Acceptable but maintenance/treatment advised)
              - `2`: UNSAFE ❌ (Exceeds contamination thresholds)
            
            #### 🛠️ How to launch local web servers:
            - **Gradio Dashboard**: `python gradio_app.py`
            - **Streamlit Web App**: `streamlit run app.py`
            - **Train Model**: `python model_training.py`
            """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
