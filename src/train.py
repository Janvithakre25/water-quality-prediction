import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.preprocessing import load_data, prepare_features_and_target, split_and_scale_data, save_scaler
from src.evaluation import evaluate_model_performance

# Ensure safe output printing on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def train_and_save_model(data_path=None, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    """
    Train machine learning models on water purification dataset, evaluate metrics,
    and save trained model and scaler.
    """
    df = load_data(data_path)
    X, y = prepare_features_and_target(df)
    
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)

    # Save scaler instance
    save_scaler(scaler, scaler_path)
    save_scaler(scaler, "scaler.pkl")

    # Define model candidate (RandomForestClassifier tuned for realistic sensor variance)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        random_state=42
    )

    print("[INFO] Training Random Forest Classifier on sensor dataset...")
    model.fit(X_train, y_train)

    # Evaluate metrics
    metrics = evaluate_model_performance(model, X_test, y_test, is_scaled=False)
    
    print("\n" + "="*50)
    print(f"[EVALUATION RESULTS]")
    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print("="*50)
    print("\n[CLASSIFICATION REPORT]:\n", metrics['classification_report'])
    print("\n[CONFUSION MATRIX]:\n", metrics['confusion_matrix'])

    # Save trained model to disk
    targets = [model_path, "model.pkl"]
    for target in targets:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(model, f)
            
    print(f"\n[SUCCESS] Model saved successfully to '{model_path}' and 'model.pkl'.")
    return model, scaler, metrics

if __name__ == "__main__":
    train_and_save_model()
