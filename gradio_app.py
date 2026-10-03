import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# ---------------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
DATASET_PATH = "water_purification_dataset.csv"

model = None
scaler = None
df_data = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

if os.path.exists(DATASET_PATH):
    df_data = pd.read_csv(DATASET_PATH)

LABEL_MAP = {0: "Safe ✅", 1: "Moderate ⚠️", 2: "Unsafe ❌"}
COLOR_MAP = {0: "#2ea44f", 1: "#d97706", 2: "#dc2626"}

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_water_quality(pH, turbidity, tds, flow_rate, pressure, temp, usage, days):
    if model is None or scaler is None:
        return "⚠️ Error: model.pkl or scaler.pkl not loaded.", "", ""

    input_feats = np.array([[pH, turbidity, tds, flow_rate, pressure, temp, usage, days]])
    scaled_feats = scaler.transform(input_feats)
    
    pred_class = model.predict(scaled_feats)[0]
    status_str = LABEL_MAP.get(pred_class, "Unknown")
    
    # Probabilities if model supports predict_proba
    probs_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_feats)[0]
        probs_dict = {
            "Safe (0)": float(probs[0]),
            "Moderate (1)": float(probs[1]),
            "Unsafe (2)": float(probs[2])
        }

    # Parameter safety recommendations
    recommendations = []
    if not (6.5 <= pH <= 8.5):
        recommendations.append(f"• **pH ({pH})**: Outside standard WHO range (6.5 - 8.5). Consider adjusting chemical balance.")
    else:
        recommendations.append(f"• **pH ({pH})**: Within safe optimal range (6.5 - 8.5).")

    if turbidity > 5.0:
        recommendations.append(f"• **Turbidity ({turbidity} NTU)**: Exceeds safe limit (5.0 NTU). High cloudiness detected; filter change recommended.")
    elif turbidity > 1.0:
        recommendations.append(f"• **Turbidity ({turbidity} NTU)**: Acceptable, but above ideal <1.0 NTU.")
    else:
        recommendations.append(f"• **Turbidity ({turbidity} NTU)**: Excellent clarity.")

    if tds > 500:
        recommendations.append(f"• **TDS ({tds} ppm)**: High total dissolved solids (>500 ppm). Consider RO membrane service.")
    elif tds > 300:
        recommendations.append(f"• **TDS ({tds} ppm)**: Fair level (300-500 ppm).")
    else:
        recommendations.append(f"• **TDS ({tds} ppm)**: Good mineral/dissolved solid level.")

    if days > 90:
        recommendations.append(f"• **Filter Usage ({days} days)**: Filter has been in use for over 90 days. Maintenance recommended.")

    rec_markdown = "### 📋 Parameter & Safety Assessment:\n" + "\n".join(recommendations)

    return status_str, probs_dict, rec_markdown

# ---------------------------------------------------------
# BATCH PREDICTION FUNCTION
# ---------------------------------------------------------
def process_batch_csv(file_obj):
    if file_obj is None:
        return None, "Please upload a valid CSV file."
    
    if model is None or scaler is None:
        return None, "Model or Scaler not loaded."

    try:
        batch_df = pd.read_csv(file_obj.name)
        required_cols = [
            "pH", "turbidity_NTU", "TDS_ppm", "flow_rate_L_min",
            "pressure_bar", "temperature_C", "usage_L_per_day", "days_since_filter_change"
        ]
        
        missing = [c for c in required_cols if c not in batch_df.columns]
        if missing:
            return None, f"Missing required columns in CSV: {missing}"

        X_batch = batch_df[required_cols]
        X_scaled = scaler.transform(X_batch)
        preds = model.predict(X_scaled)

        batch_df["Predicted_Quality_Code"] = preds
        batch_df["Predicted_Quality_Status"] = [LABEL_MAP.get(p, "Unknown") for p in preds]

        out_path = "batch_predictions_output.csv"
        batch_df.to_csv(out_path, index=False)

        return batch_df, out_path
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# ---------------------------------------------------------
# VISUALIZATION FUNCTION
# ---------------------------------------------------------
def generate_dataset_chart(chart_type):
    if df_data is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "water_purification_dataset.csv not found", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="whitegrid")

    if chart_type == "Water Quality Distribution":
        sns.countplot(data=df_data, x="water_quality", palette=["#2ea44f", "#d97706", "#dc2626"], ax=ax)
        ax.set_xticklabels(["Safe (0)", "Moderate (1)", "Unsafe (2)"])
        ax.set_title("Distribution of Water Quality Classes", fontsize=14, fontweight='bold')
        ax.set_xlabel("Water Quality Status")
        ax.set_ylabel("Count")

    elif chart_type == "TDS vs Turbidity Scatter":
        sns.scatterplot(
            data=df_data, x="TDS_ppm", y="turbidity_NTU",
            hue="water_quality", palette=["#2ea44f", "#d97706", "#dc2626"],
            alpha=0.8, ax=ax
        )
        ax.set_title("TDS (ppm) vs Turbidity (NTU) by Quality", fontsize=14, fontweight='bold')

    elif chart_type == "Correlation Heatmap":
        numeric_df = df_data.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Dataset Correlation Heatmap", fontsize=14, fontweight='bold')

    elif chart_type == "pH Distribution by Quality":
        sns.boxplot(data=df_data, x="water_quality", y="pH", palette=["#2ea44f", "#d97706", "#dc2626"], ax=ax)
        ax.set_xticklabels(["Safe (0)", "Moderate (1)", "Unsafe (2)"])
        ax.set_title("pH Range across Water Quality Classes", fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# GRADIO INTERFACE BUILD
# ---------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan"
)

with gr.Blocks(theme=custom_theme, title="💧 Smart Water Quality Prediction Dashboard") as demo:
    gr.Markdown(
        """
        # 💧 Smart Water Purification System - Interactive Dashboard
        Predict water safety status, evaluate WHO/EPA quality guidelines, conduct batch inference, and analyze dataset metrics.
        """
    )

    with gr.Tabs():
        # TAB 1: Single Prediction
        with gr.TabItem("🔮 Predict Water Quality"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Sensor Inputs")
                    pH = gr.Slider(6.0, 9.0, value=7.2, step=0.01, label="pH Level (6.0 - 9.0)")
                    turbidity = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label="Turbidity (NTU)")
                    tds = gr.Slider(100, 1000, value=300, step=1, label="TDS (ppm)")
                    flow_rate = gr.Slider(0.5, 2.0, value=1.0, step=0.01, label="Flow Rate (L/min)")
                    pressure = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Pressure (bar)")
                    temp = gr.Slider(15.0, 35.0, value=25.0, step=0.5, label="Temperature (°C)")
                    usage = gr.Slider(5, 50, value=20, step=1, label="Usage (L/day)")
                    days = gr.Slider(1, 180, value=60, step=1, label="Days Since Filter Change")
                    
                    predict_btn = gr.Button("⚡ Predict Water Quality", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Prediction Results")
                    status_output = gr.Textbox(label="Water Quality Status", interactive=False)
                    probs_output = gr.Label(label="Class Probabilities")
                    rec_output = gr.Markdown("### Assessment details will appear here after prediction.")

            predict_btn.click(
                fn=predict_water_quality,
                inputs=[pH, turbidity, tds, flow_rate, pressure, temp, usage, days],
                outputs=[status_output, probs_output, rec_output]
            )

        # TAB 2: Batch CSV Inference
        with gr.TabItem("📁 Batch Prediction (CSV)"):
            gr.Markdown("### Upload a CSV file containing sensor features to generate batch predictions.")
            with gr.Row():
                csv_input = gr.File(label="Upload Water Quality CSV Data", file_types=[".csv"])
                with gr.Column():
                    batch_btn = gr.Button("🚀 Run Batch Prediction", variant="primary")
                    status_text = gr.Textbox(label="Batch Status", interactive=False)

            batch_dataframe = gr.Dataframe(label="Batch Results Preview")
            download_file = gr.File(label="Download Predictions CSV")

            batch_btn.click(
                fn=process_batch_csv,
                inputs=[csv_input],
                outputs=[batch_dataframe, download_file]
            )

        # TAB 3: Analytics & Visualizations
        with gr.TabItem("📈 Dataset Analytics"):
            gr.Markdown("### Explore dataset trends, correlations, and parameter distributions.")
            chart_selector = gr.Dropdown(
                choices=[
                    "Water Quality Distribution",
                    "TDS vs Turbidity Scatter",
                    "Correlation Heatmap",
                    "pH Distribution by Quality"
                ],
                value="Water Quality Distribution",
                label="Select Visualization Plot"
            )
            plot_output = gr.Plot(label="Interactive Chart")

            chart_selector.change(
                fn=generate_dataset_chart,
                inputs=[chart_selector],
                outputs=[plot_output]
            )

        # TAB 4: WHO Standards Reference
        with gr.TabItem("📘 Water Quality Benchmarks"):
            gr.Markdown(
                """
                ### 🌊 Recommended Water Quality Limits (WHO / EPA)

                | Parameter | Safe Range / Limit | Importance & Impact |
                |---|---|---|
                | **pH** | 6.5 – 8.5 | Indicates acidity/alkalinity. Values < 6.5 are corrosive; > 8.5 scale-forming. |
                | **Turbidity** | < 1.0 NTU (Max 5.0) | Measures cloudiness. High turbidity harbors pathogens and clogs filters. |
                | **TDS** | < 300 - 500 ppm | Total Dissolved Solids. High TDS affects taste and filter lifespan. |
                | **Flow Rate** | 0.8 – 1.8 L/min | Optimal membrane filtration velocity. |
                | **Pressure** | 1.5 – 4.5 bar | Required RO membrane operating pressure. |
                | **Filter Replacement** | Every 90–120 days | Regular change prevents bacterial growth and membrane damage. |
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
