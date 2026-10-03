import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "pH", "turbidity_NTU", "TDS_ppm", "flow_rate_L_min",
    "pressure_bar", "temperature_C", "usage_L_per_day", "days_since_filter_change"
]
TARGET_COL = "water_quality"

def load_data(filepath=None):
    """Load dataset from provided path or fallback paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths_to_try = [
        filepath,
        os.path.join(base_dir, "data", "water_purification_dataset.csv"),
        os.path.join(base_dir, "water_purification_dataset.csv"),
        os.path.join("data", "water_purification_dataset.csv"),
        "water_purification_dataset.csv"
    ]
    for path in paths_to_try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Could not find 'water_purification_dataset.csv' in expected directories.")

def prepare_features_and_target(df):
    """Extract feature set X and target label y."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into train and test sets and scale using StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_scaler(scaler, scaler_path=None):
    """Save fitted scaler to disk."""
    targets = [
        scaler_path,
        os.path.join("models", "scaler.pkl"),
        "scaler.pkl"
    ]
    for target in targets:
        if target:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with open(target, "wb") as f:
                pickle.dump(scaler, f)

def load_scaler(scaler_path=None):
    """Load fitted scaler from disk."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = [
        scaler_path,
        os.path.join(base_dir, "models", "scaler.pkl"),
        os.path.join(base_dir, "scaler.pkl"),
        os.path.join("models", "scaler.pkl"),
        "scaler.pkl"
    ]
    for target in targets:
        if target and os.path.exists(target):
            with open(target, "rb") as f:
                return pickle.load(f)
    return None
