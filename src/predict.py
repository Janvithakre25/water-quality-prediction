import os
import pickle
import numpy as np
import pandas as pd
from src.preprocessing import FEATURE_COLS, load_scaler

LABEL_MAP = {0: "Safe ✅", 1: "Moderate ⚠️", 2: "Unsafe ❌"}

def load_trained_model(model_path=None):
    """Load model pickle file from provided or default path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths_to_try = [
        model_path,
        os.path.join(base_dir, "models", "model.pkl"),
        os.path.join(base_dir, "model.pkl"),
        os.path.join("models", "model.pkl"),
        "model.pkl"
    ]
    for p in paths_to_try:
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None

def predict_single_sample(input_values, model=None, scaler=None):
    """
    Predict water quality class for single sample input dict, dataframe, or list.
    Preserves feature names to prevent scikit-learn UserWarnings.
    """
    if model is None:
        model = load_trained_model()
    if scaler is None:
        scaler = load_scaler()

    if model is None or scaler is None:
        raise FileNotFoundError("Model or Scaler binary files not found.")

    if isinstance(input_values, dict):
        input_df = pd.DataFrame([[input_values[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    elif isinstance(input_values, (list, np.ndarray)):
        arr = np.array(input_values).reshape(1, -1)
        input_df = pd.DataFrame(arr, columns=FEATURE_COLS)
    elif isinstance(input_values, pd.DataFrame):
        input_df = input_values[FEATURE_COLS]
    else:
        raise ValueError("Invalid input format for single sample prediction.")

    # Predict
    pred_code = int(model.predict(input_df)[0])
    status_label = LABEL_MAP.get(pred_code, "Unknown")
    
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        probabilities = {
            "Safe (0)": float(probs[0]),
            "Moderate (1)": float(probs[1]),
            "Unsafe (2)": float(probs[2])
        }

    return pred_code, status_label, probabilities

def predict_batch_df(df, model=None, scaler=None):
    """
    Predict water quality classes for a dataframe of sensor samples.
    """
    if model is None:
        model = load_trained_model()
    if scaler is None:
        scaler = load_scaler()

    if model is None or scaler is None:
        raise FileNotFoundError("Model or Scaler binary files not found.")

    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    X_batch = df[FEATURE_COLS]
    predictions = model.predict(X_batch)
    
    out_df = df.copy()
    out_df["Predicted_Quality_Code"] = predictions
    out_df["Predicted_Quality_Status"] = [LABEL_MAP.get(p, "Unknown") for p in predictions]

    return out_df
