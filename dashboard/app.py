import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from src.preprocessing import FEATURE_COLS, load_data, load_scaler, split_and_scale_data
from src.predict import predict_single_sample, predict_batch_df, load_trained_model, LABEL_MAP
from src.evaluation import evaluate_model_performance

# ---------------------------------------------------------
# INITIALIZE MODEL & DATASET
# ---------------------------------------------------------
model = load_trained_model()
scaler = load_scaler()
try:
    df_dataset = load_data()
except Exception:
    df_dataset = None

DEFAULT_INPUTS = {
    "pH": 7.2,
    "turbidity_NTU": 2.0,
    "TDS_ppm": 300,
    "flow_rate_L_min": 1.0,
    "pressure_bar": 2.5,
    "temperature_C": 25.0,
    "usage_L_per_day": 20,
    "days_since_filter_change": 60
}

# Set aesthetic plot style
plt.style.use("ggplot")
sns.set_theme(style="darkgrid")

# ---------------------------------------------------------
# LOGIC & FORMATTING FUNCTIONS
# ---------------------------------------------------------
def predict_single_formatted(pH, turbidity, tds, flow, pressure, temp, usage, days):
    try:
        sample_dict = {
            "pH": pH, "turbidity_NTU": turbidity, "TDS_ppm": tds,
            "flow_rate_L_min": flow, "pressure_bar": pressure,
            "temperature_C": temp, "usage_L_per_day": usage,
            "days_since_filter_change": days
        }
        pred_code, status_label, probs = predict_single_sample(sample_dict, model, scaler)
        
        # HTML Badge formatting for status
        if pred_code == 0:
            badge_html = """
            <div style="background-color: #10B981; color: white; padding: 16px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 10px;">
                ✅ Water Quality: SAFE
            </div>
            """
        elif pred_code == 1:
            badge_html = """
            <div style="background-color: #F59E0B; color: white; padding: 16px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 10px;">
                ⚠️ Water Quality: MODERATE
            </div>
            """
        else:
            badge_html = """
            <div style="background-color: #EF4444; color: white; padding: 16px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 10px;">
                ❌ Water Quality: UNSAFE
            </div>
            """

        # Detailed assessment items
        recs = []
        if 6.5 <= pH <= 8.5:
            recs.append("<li style='color:#10B981;'><b>pH Level (<b>{}</b>)</b>: Optimal & Safe (6.5 – 8.5 WHO standard).</li>".format(pH))
        else:
            recs.append("<li style='color:#EF4444;'><b>pH Level (<b>{}</b>)</b>: Outside safe WHO limit (6.5 – 8.5). Adjust chemical dosage.</li>".format(pH))

        if turbidity <= 1.0:
            recs.append("<li style='color:#10B981;'><b>Turbidity (<b>{} NTU</b>)</b>: Excellent optical clarity (&lt; 1.0 NTU).</li>".format(turbidity))
        elif turbidity <= 5.0:
            recs.append("<li style='color:#F59E0B;'><b>Turbidity (<b>{} NTU</b>)</b>: Acceptable clarity (&lt; 5.0 NTU).</li>".format(turbidity))
        else:
            recs.append("<li style='color:#EF4444;'><b>Turbidity (<b>{} NTU</b>)</b>: High turbidity (&gt; 5.0 NTU). High cloudiness; filter service needed.</li>".format(turbidity))

        if tds <= 300:
            recs.append("<li style='color:#10B981;'><b>TDS (<b>{} ppm</b>)</b>: Ideal mineral dissolved solids (&lt; 300 ppm).</li>".format(tds))
        elif tds <= 500:
            recs.append("<li style='color:#F59E0B;'><b>TDS (<b>{} ppm</b>)</b>: Fair mineral levels (300 – 500 ppm).</li>".format(tds))
        else:
            recs.append("<li style='color:#EF4444;'><b>TDS (<b>{} ppm</b>)</b>: Elevated dissolved solids (&gt; 500 ppm). RO membrane service advised.</li>".format(tds))

        if days > 90:
            recs.append("<li style='color:#F59E0B;'><b>Filter Lifecycle (<b>{} days</b>)</b>: Filter in service over 90 days. Maintenance recommended.</li>".format(days))
        else:
            recs.append("<li style='color:#10B981;'><b>Filter Lifecycle (<b>{} days</b>)</b>: Filter within normal operational timeframe.</li>".format(days))

        rec_html = "<div style='padding: 12px; border: 1px solid #374151; border-radius: 8px; font-size: 15px;'><h4 style='margin-top:0;'>📋 Parameter Safety Evaluation</h4><ul>" + "".join(recs) + "</ul></div>"

        return badge_html, probs, rec_html
    except Exception as e:
        err_html = f"<div style='color:red;'>Error: {str(e)}</div>"
        return err_html, {}, "Error computing evaluation."

def reset_single_inputs():
    default_badge = """
    <div style="background-color: #3B82F6; color: white; padding: 16px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold;">
        💧 Ready for Input Evaluation
    </div>
    """
    return (
        DEFAULT_INPUTS["pH"],
        DEFAULT_INPUTS["turbidity_NTU"],
        DEFAULT_INPUTS["TDS_ppm"],
        DEFAULT_INPUTS["flow_rate_L_min"],
        DEFAULT_INPUTS["pressure_bar"],
        DEFAULT_INPUTS["temperature_C"],
        DEFAULT_INPUTS["usage_L_per_day"],
        DEFAULT_INPUTS["days_since_filter_change"],
        default_badge,
        {},
        "<i>Click 'Predict Quality' to generate assessment.</i>"
    )

def process_batch(file_obj):
    if file_obj is None:
        return None, None, "⚠️ Please upload a valid CSV dataset file."
    try:
        input_df = pd.read_csv(file_obj.name)
        result_df = predict_batch_df(input_df, model, scaler)
        
        out_path = "batch_predictions_output.csv"
        result_df.to_csv(out_path, index=False)

        counts = result_df["Predicted_Quality_Status"].value_counts().to_dict()
        summary_str = f"✅ Processed {len(result_df)} samples successfully. Quality breakdown: {counts}"
        return result_df, out_path, summary_str
    except Exception as e:
        return None, None, f"❌ Error processing CSV: {str(e)}"

def render_evaluation_plots():
    if df_dataset is None or model is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Dataset or Model unavailable", ha='center')
        return fig, fig

    X = df_dataset[FEATURE_COLS]
    y = df_dataset["water_quality"]
    _, X_test, _, _, _, y_test, _ = split_and_scale_data(X, y)
    metrics = evaluate_model_performance(model, X_test, y_test)

    # Plot 1: Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5), dpi=100)
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=["Safe", "Moderate", "Unsafe"],
                yticklabels=["Safe", "Moderate", "Unsafe"], ax=ax_cm, cbar=False)
    ax_cm.set_title("Validation Confusion Matrix (240 Test Samples)", fontsize=12, fontweight="bold", pad=12)
    ax_cm.set_ylabel("Actual Class", fontsize=10, fontweight="bold")
    ax_cm.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    plt.tight_layout()

    # Plot 2: Feature Importance
    fig_imp, ax_imp = plt.subplots(figsize=(7, 4.5), dpi=100)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        sorted_feats = [FEATURE_COLS[i] for i in indices]
        sorted_imps = importances[indices]
        
        sns.barplot(x=sorted_imps, y=sorted_feats, palette="crest", ax=ax_imp)
        ax_imp.set_title("Decision Tree Relative Feature Importance", fontsize=12, fontweight="bold", pad=12)
        ax_imp.set_xlabel("Importance Score", fontsize=10, fontweight="bold")
    plt.tight_layout()

    return fig_cm, fig_imp

def render_visualization_plot(chart_type):
    if df_dataset is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Dataset unavailable", ha='center')
        return fig

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=100)

    if chart_type == "Class Distribution":
        sns.countplot(data=df_dataset, x="water_quality", palette=["#10B981", "#F59E0B", "#EF4444"], ax=ax)
        ax.set_xticklabels(["Safe (0)", "Moderate (1)", "Unsafe (2)"])
        ax.set_title("Water Quality Class Counts in Dataset", fontsize=12, fontweight="bold")
        ax.set_xlabel("Quality Category", fontweight="bold")
        ax.set_ylabel("Sample Count", fontweight="bold")

    elif chart_type == "TDS vs Turbidity Scatter":
        sns.scatterplot(data=df_dataset, x="TDS_ppm", y="turbidity_NTU", hue="water_quality",
                        palette=["#10B981", "#F59E0B", "#EF4444"], alpha=0.8, ax=ax)
        ax.set_title("TDS (ppm) vs Turbidity (NTU) Scatter by Quality Class", fontsize=12, fontweight="bold")
        ax.set_xlabel("TDS (ppm)", fontweight="bold")
        ax.set_ylabel("Turbidity (NTU)", fontweight="bold")

    elif chart_type == "Correlation Heatmap":
        numeric_df = df_dataset.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax, cbar=False)
        ax.set_title("Sensor Feature Correlation Matrix", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# GRADIO UI DASHBOARD BUILD
# ---------------------------------------------------------
def create_gradio_dashboard():
    css = """
    .main-title { text-align: center; font-size: 40px; font-weight: bold; margin-bottom: 5px; }
    .sub-title { text-align: center; font-size: 15px; color: #9CA3AF; margin-bottom: 20px; }
    .kpi-card { background-color: #1F2937; border: 1px solid #374151; padding: 15px; border-radius: 10px; text-align: center; }
    .kpi-val { font-size: 26px; font-weight: bold; color: #3B82F6; }
    .kpi-label { font-size: 13px; color: #9CA3AF; }
    """

    with gr.Blocks(title="Smart Water Quality Prediction System") as demo:
        gr.HTML(
            """
            <div class="main-title">💧 Smart Water Quality Prediction & Monitoring Dashboard</div>
            <div class="sub-title">Real-time Machine Learning Water Safety Classification, Batch Analytics, and WHO Compliance Benchmarks</div>
            """
        )

        with gr.Tabs():
            # TAB 1: Single Prediction
            with gr.TabItem("🔮 Single Sample Predictor"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Water Sensor Parameters")
                        
                        with gr.Group():
                            gr.Markdown("#### Physical Properties")
                            pH = gr.Slider(6.0, 9.0, value=7.2, step=0.01, label="pH Level (6.0 - 9.0)")
                            turbidity = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label="Turbidity (0.1 - 10.0 NTU)")
                            tds = gr.Slider(100, 1000, value=300, step=1, label="TDS (100 - 1000 ppm)")
                            temp = gr.Slider(15.0, 35.0, value=25.0, step=0.5, label="Temperature (15 - 35 °C)")

                        with gr.Group():
                            gr.Markdown("#### Operational Parameters")
                            flow = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Flow Rate (0.5 - 2.0 L/min)")
                            pressure = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Pressure (1.0 - 5.0 bar)")
                            usage = gr.Slider(5, 50, value=20, step=1, label="Daily Usage (5 - 50 L/day)")
                            days = gr.Slider(1, 180, value=60, step=1, label="Days Since Filter Change (1 - 180 days)")

                        with gr.Row():
                            predict_btn = gr.Button("⚡ Predict Water Quality", variant="primary", scale=2)
                            reset_btn = gr.Button("🔄 Reset", scale=1)

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Prediction & Assessment")
                        status_html = gr.HTML(
                            """
                            <div style="background-color: #3B82F6; color: white; padding: 16px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold;">
                                💧 Ready for Input Evaluation
                            </div>
                            """
                        )
                        probs_out = gr.Label(label="Class Probabilities Breakdown")
                        rec_html = gr.HTML("<i>Adjust parameters and click 'Predict Water Quality' to view evaluation.</i>")

                predict_btn.click(
                    fn=predict_single_formatted,
                    inputs=[pH, turbidity, tds, flow, pressure, temp, usage, days],
                    outputs=[status_html, probs_out, rec_html]
                )

                reset_btn.click(
                    fn=reset_single_inputs,
                    outputs=[pH, turbidity, tds, flow, pressure, temp, usage, days, status_html, probs_out, rec_html]
                )

            # TAB 2: Batch CSV Prediction
            with gr.TabItem("📁 Batch CSV Processing"):
                gr.Markdown("### Upload a CSV dataset containing sensor readings to predict water quality for multiple samples.")
                with gr.Row():
                    csv_in = gr.File(label="Upload Water Sensor CSV File", file_types=[".csv"])
                    with gr.Column():
                        batch_btn = gr.Button("🚀 Process Batch Predictions", variant="primary")
                        batch_status = gr.Textbox(label="Batch Execution Status", interactive=False)

                batch_table = gr.Dataframe(label="Results Data Table Preview")
                batch_file_out = gr.File(label="Download Processed CSV File")

                batch_btn.click(
                    fn=process_batch,
                    inputs=[csv_in],
                    outputs=[batch_table, batch_file_out, batch_status]
                )

            # TAB 3: Model Performance Metrics
            with gr.TabItem("📈 Model Evaluation & KPI Metrics"):
                gr.Markdown("### Empirical Model Validation Metrics (Tuned Decision Tree Classifier)")
                
                gr.HTML(
                    """
                    <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                        <div class="kpi-card" style="flex: 1;">
                            <div class="kpi-val">97.08%</div>
                            <div class="kpi-label">Test Accuracy</div>
                        </div>
                        <div class="kpi-card" style="flex: 1;">
                            <div class="kpi-val">97.11%</div>
                            <div class="kpi-label">Weighted Precision</div>
                        </div>
                        <div class="kpi-card" style="flex: 1;">
                            <div class="kpi-val">97.08%</div>
                            <div class="kpi-label">Weighted Recall</div>
                        </div>
                        <div class="kpi-card" style="flex: 1;">
                            <div class="kpi-val">97.08%</div>
                            <div class="kpi-label">Weighted F1-Score</div>
                        </div>
                    </div>
                    """
                )
                
                cm_btn = gr.Button("📊 Load Model Performance Charts", variant="primary")
                with gr.Row():
                    cm_plot = gr.Plot(label="Confusion Matrix Heatmap")
                    imp_plot = gr.Plot(label="Feature Importances Bar Chart")

                cm_btn.click(
                    fn=render_evaluation_plots,
                    outputs=[cm_plot, imp_plot]
                )

            # TAB 4: Dataset Exploration
            with gr.TabItem("📊 Dataset Exploration & Charts"):
                gr.Markdown("### Explore Dataset Trends and Sensor Distributions")
                if df_dataset is not None:
                    gr.Dataframe(value=df_dataset.head(10), label="Dataset Preview (First 10 Rows)")
                
                chart_dropdown = gr.Dropdown(
                    choices=["Class Distribution", "TDS vs Turbidity Scatter", "Correlation Heatmap"],
                    value="Class Distribution",
                    label="Select Visualization Plot"
                )
                data_plot = gr.Plot(label="Dataset Visualization Plot")

                chart_dropdown.change(
                    fn=render_visualization_plot,
                    inputs=[chart_dropdown],
                    outputs=[data_plot]
                )

    return demo

def launch_dashboard_app(demo, start_port=7860, max_port=7875, host="127.0.0.1", share=False):
    """
    Launches Gradio app, trying start_port up to max_port automatically
    if the initial port is occupied. Defaults to share=False (local-only).
    """
    for port in range(start_port, max_port + 1):
        try:
            print(f"[INFO] Attempting to launch Gradio dashboard on http://{host}:{port} (share={share})...")
            res = demo.launch(
                server_name=host,
                server_port=port,
                share=share,
                prevent_thread_lock=False
            )
            print(f"[SUCCESS] Gradio dashboard running on http://{host}:{port}")
            return res
        except OSError as e:
            err_msg = str(e).lower()
            if "port" in err_msg or "address already in use" in err_msg or "cannot find empty port" in err_msg:
                print(f"[WARNING] Port {port} is occupied. Retrying with port {port + 1}...")
                continue
            raise e
    raise OSError(f"Could not find an available port in range {start_port}–{max_port}.")

if __name__ == "__main__":
    app = create_gradio_dashboard()
    launch_dashboard_app(app, start_port=7860, max_port=7875, host="127.0.0.1", share=False)
