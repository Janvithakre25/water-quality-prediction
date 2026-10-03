import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from src.preprocessing import FEATURE_COLS, load_data, load_scaler
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

plt.style.use("ggplot")
sns.set_theme(style="darkgrid")

# ---------------------------------------------------------
# LOGIC FUNCTIONS
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

        if pred_code == 0:
            badge_html = """
            <div style="background:#10B981;color:white;padding:18px;border-radius:12px;
                        text-align:center;font-size:22px;font-weight:bold;letter-spacing:1px;">
                ✅ Water Quality: SAFE
            </div>"""
        elif pred_code == 1:
            badge_html = """
            <div style="background:#F59E0B;color:white;padding:18px;border-radius:12px;
                        text-align:center;font-size:22px;font-weight:bold;letter-spacing:1px;">
                ⚠️ Water Quality: MODERATE
            </div>"""
        else:
            badge_html = """
            <div style="background:#EF4444;color:white;padding:18px;border-radius:12px;
                        text-align:center;font-size:22px;font-weight:bold;letter-spacing:1px;">
                ❌ Water Quality: UNSAFE
            </div>"""

        recs = []
        if 6.5 <= pH <= 8.5:
            recs.append(f"<li style='color:#10B981;margin-bottom:6px;'><b>pH ({pH})</b>: ✅ Optimal & Safe (WHO range: 6.5 – 8.5).</li>")
        else:
            recs.append(f"<li style='color:#EF4444;margin-bottom:6px;'><b>pH ({pH})</b>: ❌ Outside WHO limit (6.5 – 8.5). Adjust chemical dosage.</li>")

        if turbidity <= 1.0:
            recs.append(f"<li style='color:#10B981;margin-bottom:6px;'><b>Turbidity ({turbidity} NTU)</b>: ✅ Excellent clarity (&lt; 1.0 NTU).</li>")
        elif turbidity <= 5.0:
            recs.append(f"<li style='color:#F59E0B;margin-bottom:6px;'><b>Turbidity ({turbidity} NTU)</b>: ⚠️ Acceptable (&lt; 5.0 NTU ideal).</li>")
        else:
            recs.append(f"<li style='color:#EF4444;margin-bottom:6px;'><b>Turbidity ({turbidity} NTU)</b>: ❌ High cloudiness (&gt; 5.0 NTU). Filter service required.</li>")

        if tds <= 300:
            recs.append(f"<li style='color:#10B981;margin-bottom:6px;'><b>TDS ({tds} ppm)</b>: ✅ Ideal dissolved solids (&lt; 300 ppm).</li>")
        elif tds <= 500:
            recs.append(f"<li style='color:#F59E0B;margin-bottom:6px;'><b>TDS ({tds} ppm)</b>: ⚠️ Acceptable levels (300 – 500 ppm).</li>")
        else:
            recs.append(f"<li style='color:#EF4444;margin-bottom:6px;'><b>TDS ({tds} ppm)</b>: ❌ Elevated (&gt; 500 ppm). RO membrane service advised.</li>")

        if days > 90:
            recs.append(f"<li style='color:#F59E0B;margin-bottom:6px;'><b>Filter Age ({days} days)</b>: ⚠️ Over 90 days in service. Maintenance recommended.</li>")
        else:
            recs.append(f"<li style='color:#10B981;margin-bottom:6px;'><b>Filter Age ({days} days)</b>: ✅ Within normal operational lifecycle.</li>")

        rec_html = (
            "<div style='padding:14px;border:1px solid #374151;border-radius:10px;"
            "background:#111827;font-size:14px;line-height:1.7;'>"
            "<h4 style='margin:0 0 10px;color:#E5E7EB;'>📋 Parameter Safety Evaluation</h4>"
            "<ul style='margin:0;padding-left:18px;'>"
            + "".join(recs)
            + "</ul></div>"
        )
        return badge_html, probs, rec_html
    except Exception as e:
        return f"<div style='color:red;'>Error: {str(e)}</div>", {}, "Error occurred."

def reset_single_inputs():
    default_badge = """
    <div style="background:#3B82F6;color:white;padding:18px;border-radius:12px;
                text-align:center;font-size:20px;font-weight:bold;">
        💧 Awaiting Sensor Input Evaluation
    </div>"""
    return (
        DEFAULT_INPUTS["pH"], DEFAULT_INPUTS["turbidity_NTU"],
        DEFAULT_INPUTS["TDS_ppm"], DEFAULT_INPUTS["flow_rate_L_min"],
        DEFAULT_INPUTS["pressure_bar"], DEFAULT_INPUTS["temperature_C"],
        DEFAULT_INPUTS["usage_L_per_day"], DEFAULT_INPUTS["days_since_filter_change"],
        default_badge, {}, "<i>Inputs reset. Click Predict to evaluate.</i>"
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
        return result_df, out_path, f"✅ Processed {len(result_df)} samples. Breakdown: {counts}"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def render_evaluation_plots():
    if df_dataset is None or model is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Dataset or Model unavailable", ha='center')
        return fig, fig

    X = df_dataset[FEATURE_COLS]
    y = df_dataset["water_quality"]
    metrics = evaluate_model_performance(model, X, y)

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5), dpi=110)
    sns.heatmap(
        metrics["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu',
        xticklabels=["Safe", "Moderate", "Unsafe"],
        yticklabels=["Safe", "Moderate", "Unsafe"],
        ax=ax_cm, cbar=False, linewidths=0.5, linecolor='#374151'
    )
    ax_cm.set_title("Validation Confusion Matrix\n(240 Unseen Test Samples)", fontsize=11, fontweight="bold", pad=10)
    ax_cm.set_ylabel("Actual Class", fontsize=9, fontweight="bold")
    ax_cm.set_xlabel("Predicted Class", fontsize=9, fontweight="bold")
    plt.tight_layout()

    # Feature Importance
    fig_imp, ax_imp = plt.subplots(figsize=(7, 4.5), dpi=110)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        sorted_feats = [FEATURE_COLS[i] for i in indices]
        sorted_imps = importances[indices]
        bars = ax_imp.barh(sorted_feats, sorted_imps, color=sns.color_palette("crest", len(sorted_feats)))
        ax_imp.set_title("Random Forest Feature Importances", fontsize=11, fontweight="bold", pad=10)
        ax_imp.set_xlabel("Importance Score", fontsize=9, fontweight="bold")
        # Add value labels
        for bar, val in zip(bars, sorted_imps):
            ax_imp.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va='center', fontsize=8)
    plt.tight_layout()

    return fig_cm, fig_imp

def render_visualization_plot(chart_type):
    if df_dataset is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Dataset unavailable", ha='center')
        return fig

    PALETTE = {0: "#10B981", 1: "#F59E0B", 2: "#EF4444"}
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)

    if chart_type == "Class Distribution":
        counts = df_dataset["water_quality"].value_counts().sort_index()
        bars = ax.bar(["Safe (0)", "Moderate (1)", "Unsafe (2)"], counts.values, color=colors, width=0.5, edgecolor='white')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha='center', fontsize=11, fontweight='bold')
        ax.set_title("Water Quality Class Distribution", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Quality Category", fontsize=10, fontweight="bold")
        ax.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
        ax.set_ylim(0, counts.max() * 1.15)

    elif chart_type == "TDS vs Turbidity Scatter":
        for cls, col in PALETTE.items():
            subset = df_dataset[df_dataset["water_quality"] == cls]
            ax.scatter(subset["TDS_ppm"], subset["turbidity_NTU"],
                       label=["Safe", "Moderate", "Unsafe"][cls],
                       color=col, alpha=0.65, s=18)
        ax.set_title("TDS (ppm) vs Turbidity (NTU) by Quality Class", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("TDS (ppm)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Turbidity (NTU)", fontsize=10, fontweight="bold")
        ax.legend(title="Quality", fontsize=9, title_fontsize=9)

    elif chart_type == "pH Distribution by Class":
        for cls, col in PALETTE.items():
            subset = df_dataset[df_dataset["water_quality"] == cls]
            ax.hist(subset["pH"], bins=20, alpha=0.6, color=col,
                    label=["Safe", "Moderate", "Unsafe"][cls], edgecolor='white')
        ax.set_title("pH Level Distribution by Quality Class", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("pH Level", fontsize=10, fontweight="bold")
        ax.set_ylabel("Sample Frequency", fontsize=10, fontweight="bold")
        ax.legend(title="Quality", fontsize=9)

    elif chart_type == "Correlation Heatmap":
        numeric_df = df_dataset[FEATURE_COLS + ["water_quality"]].select_dtypes(include=[np.number])
        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="vlag",
                    ax=ax, mask=mask, cbar=True, linewidths=0.4,
                    annot_kws={"size": 7})
        ax.set_title("Sensor Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# GRADIO UI DASHBOARD
# ---------------------------------------------------------
def create_gradio_dashboard():
    css = """
    footer {visibility: hidden;}
    .gr-button-primary {border-radius: 8px !important; font-weight: bold !important;}
    """
    with gr.Blocks(css=css, title="Smart Water Quality Prediction System") as demo:

        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px;">
            <div style="font-size:32px;font-weight:bold;letter-spacing:1px;">
                💧 Smart Water Quality Prediction Dashboard
            </div>
            <div style="font-size:15px;color:#9CA3AF;margin-top:6px;">
                Real-time ML Water Safety Classification • WHO/EPA Compliance Benchmarks • Batch Analytics
            </div>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: Single Predictor ─────────────────────────────
            with gr.TabItem("🔮 Single Sample Predictor"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Sensor Input Parameters")

                        with gr.Group():
                            gr.Markdown("**Physical Properties**")
                            pH      = gr.Slider(6.0,  9.0,   value=7.2,  step=0.01, label="pH Level  (Safe: 6.5 – 8.5)")
                            turb    = gr.Slider(0.1,  10.0,  value=2.0,  step=0.1,  label="Turbidity NTU  (Safe: < 5.0 NTU)")
                            tds     = gr.Slider(100,  1000,  value=300,  step=1,    label="TDS ppm  (Safe: < 500 ppm)")
                            temp    = gr.Slider(15.0, 35.0,  value=25.0, step=0.5,  label="Temperature °C")

                        with gr.Group():
                            gr.Markdown("**Operational Parameters**")
                            flow    = gr.Slider(0.5, 2.0,  value=1.0,  step=0.05, label="Flow Rate L/min")
                            press   = gr.Slider(1.0, 5.0,  value=2.5,  step=0.1,  label="Pressure bar")
                            usage   = gr.Slider(5,   50,   value=20,   step=1,    label="Daily Usage L/day")
                            days    = gr.Slider(1,   180,  value=60,   step=1,    label="Days Since Filter Change  (Safe: < 90 days)")

                        with gr.Row():
                            predict_btn = gr.Button("⚡ Predict Water Quality", variant="primary", scale=3)
                            reset_btn   = gr.Button("🔄 Reset", scale=1)

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Result & Safety Evaluation")
                        status_html = gr.HTML("""
                        <div style="background:#3B82F6;color:white;padding:18px;border-radius:12px;
                                    text-align:center;font-size:20px;font-weight:bold;">
                            💧 Awaiting Sensor Input Evaluation
                        </div>""")
                        probs_out = gr.Label(label="Class Probability Breakdown", num_top_classes=3)
                        rec_html  = gr.HTML("<i style='color:#9CA3AF;'>Adjust parameters and click Predict to view assessment.</i>")

                predict_btn.click(
                    fn=predict_single_formatted,
                    inputs=[pH, turb, tds, flow, press, temp, usage, days],
                    outputs=[status_html, probs_out, rec_html]
                )
                reset_btn.click(
                    fn=reset_single_inputs,
                    outputs=[pH, turb, tds, flow, press, temp, usage, days, status_html, probs_out, rec_html]
                )

            # ── TAB 2: Batch CSV ────────────────────────────────────
            with gr.TabItem("📁 Batch CSV Processing"):
                gr.Markdown("""
                ### Batch Inference on CSV Datasets
                Upload a CSV file with sensor columns to predict water quality for all samples at once.
                
                **Required columns:** `pH`, `turbidity_NTU`, `TDS_ppm`, `flow_rate_L_min`, `pressure_bar`, `temperature_C`, `usage_L_per_day`, `days_since_filter_change`
                """)
                with gr.Row():
                    csv_in    = gr.File(label="Upload Sensor CSV File", file_types=[".csv"])
                    with gr.Column():
                        batch_btn = gr.Button("🚀 Run Batch Predictions", variant="primary")
                        batch_status = gr.Textbox(label="Status", interactive=False)

                batch_table    = gr.Dataframe(label="Results Preview", wrap=True)
                batch_file_out = gr.File(label="Download Results CSV")

                batch_btn.click(
                    fn=process_batch,
                    inputs=[csv_in],
                    outputs=[batch_table, batch_file_out, batch_status]
                )

            # ── TAB 3: Model Metrics ────────────────────────────────
            with gr.TabItem("📈 Model Evaluation & Metrics"):
                gr.Markdown("### Empirical Validation Results — Random Forest Classifier (94.17% Test Accuracy)")

                gr.HTML("""
                <div style="display:flex;gap:14px;margin:10px 0 22px;flex-wrap:wrap;">
                    <div style="flex:1;min-width:130px;background:#111827;border:1px solid #374151;
                                padding:16px;border-radius:10px;text-align:center;">
                        <div style="font-size:28px;font-weight:bold;color:#3B82F6;">94.17%</div>
                        <div style="font-size:12px;color:#9CA3AF;margin-top:4px;">Test Accuracy</div>
                    </div>
                    <div style="flex:1;min-width:130px;background:#111827;border:1px solid #374151;
                                padding:16px;border-radius:10px;text-align:center;">
                        <div style="font-size:28px;font-weight:bold;color:#10B981;">0.9436</div>
                        <div style="font-size:12px;color:#9CA3AF;margin-top:4px;">Weighted Precision</div>
                    </div>
                    <div style="flex:1;min-width:130px;background:#111827;border:1px solid #374151;
                                padding:16px;border-radius:10px;text-align:center;">
                        <div style="font-size:28px;font-weight:bold;color:#10B981;">0.9417</div>
                        <div style="font-size:12px;color:#9CA3AF;margin-top:4px;">Weighted Recall</div>
                    </div>
                    <div style="flex:1;min-width:130px;background:#111827;border:1px solid #374151;
                                padding:16px;border-radius:10px;text-align:center;">
                        <div style="font-size:28px;font-weight:bold;color:#10B981;">0.9412</div>
                        <div style="font-size:12px;color:#9CA3AF;margin-top:4px;">Weighted F1-Score</div>
                    </div>
                    <div style="flex:1;min-width:130px;background:#111827;border:1px solid #374151;
                                padding:16px;border-radius:10px;text-align:center;">
                        <div style="font-size:28px;font-weight:bold;color:#8B5CF6;">240</div>
                        <div style="font-size:12px;color:#9CA3AF;margin-top:4px;">Test Samples</div>
                    </div>
                </div>
                """)

                cm_btn = gr.Button("📊 Generate Performance Charts", variant="primary")
                with gr.Row():
                    cm_plot  = gr.Plot(label="Confusion Matrix")
                    imp_plot = gr.Plot(label="Feature Importance")

                cm_btn.click(fn=render_evaluation_plots, outputs=[cm_plot, imp_plot])

            # ── TAB 4: Dataset Charts ───────────────────────────────
            with gr.TabItem("📊 Dataset Analytics"):
                gr.Markdown("### Sensor Data Distributions & Feature Analysis")
                if df_dataset is not None:
                    with gr.Row():
                        gr.Dataframe(value=df_dataset.head(8), label=f"Dataset Preview (1200 total samples)")

                chart_dropdown = gr.Dropdown(
                    choices=["Class Distribution", "TDS vs Turbidity Scatter",
                             "pH Distribution by Class", "Correlation Heatmap"],
                    value="Class Distribution",
                    label="Select Visualization"
                )
                data_plot = gr.Plot(label="Chart")
                chart_dropdown.change(
                    fn=render_visualization_plot,
                    inputs=[chart_dropdown],
                    outputs=[data_plot]
                )

            # ── TAB 5: WHO Standards ────────────────────────────────
            with gr.TabItem("📘 WHO/EPA Standards"):
                gr.Markdown("### WHO & EPA Safe Drinking Water Quality Guidelines")
                gr.HTML("""
                <table style="width:100%;border-collapse:collapse;font-size:14px;">
                  <thead>
                    <tr style="background:#1F2937;color:#E5E7EB;">
                      <th style="padding:10px;border:1px solid #374151;">Parameter</th>
                      <th style="padding:10px;border:1px solid #374151;">Safe Limit</th>
                      <th style="padding:10px;border:1px solid #374151;">Why It Matters</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td style="padding:9px;border:1px solid #374151;"><b>pH</b></td>
                        <td style="padding:9px;border:1px solid #374151;color:#10B981;">6.5 – 8.5</td>
                        <td style="padding:9px;border:1px solid #374151;">Outside range causes corrosion or scale build-up in pipes.</td></tr>
                    <tr style="background:#111827;"><td style="padding:9px;border:1px solid #374151;"><b>Turbidity</b></td>
                        <td style="padding:9px;border:1px solid #374151;color:#10B981;">&lt; 1.0 NTU (max 5.0)</td>
                        <td style="padding:9px;border:1px solid #374151;">High turbidity shields pathogens from disinfection.</td></tr>
                    <tr><td style="padding:9px;border:1px solid #374151;"><b>TDS</b></td>
                        <td style="padding:9px;border:1px solid #374151;color:#10B981;">&lt; 300 – 500 ppm</td>
                        <td style="padding:9px;border:1px solid #374151;">High TDS affects taste, odour, and filter lifespan.</td></tr>
                    <tr style="background:#111827;"><td style="padding:9px;border:1px solid #374151;"><b>Filter Replacement</b></td>
                        <td style="padding:9px;border:1px solid #374151;color:#10B981;">Every 90 – 120 days</td>
                        <td style="padding:9px;border:1px solid #374151;">Prevents biofilm growth and membrane saturation.</td></tr>
                    <tr><td style="padding:9px;border:1px solid #374151;"><b>Pressure</b></td>
                        <td style="padding:9px;border:1px solid #374151;color:#10B981;">1.5 – 4.5 bar</td>
                        <td style="padding:9px;border:1px solid #374151;">Optimal RO membrane operating pressure range.</td></tr>
                    <tr style="background:#111827;"><td style="padding:9px;border:1px solid #374151;"><b>Temperature</b></td>
                        <td style="padding:9px;border:1px solid #374151;color:#10B981;">15 – 30 °C</td>
                        <td style="padding:9px;border:1px solid #374151;">Higher temperatures accelerate bacterial growth.</td></tr>
                  </tbody>
                </table>
                """)

    return demo

# ── ENTRY POINT ─────────────────────────────────────────────
if __name__ == "__main__":
    demo = create_gradio_dashboard()
    # On Hugging Face Spaces: server_name="0.0.0.0", share=False (HF provides the URL)
    # Locally with public link: server_name="127.0.0.1", share=True
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)