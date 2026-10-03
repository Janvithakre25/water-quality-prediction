import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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
    Train machine learning candidate models on water purification dataset,
    select best model (Decision Tree), perform hyperparameter tuning, evaluate metrics,
    and save trained tuned model and scaler.
    """
    df = load_data(data_path)
    X, y = prepare_features_and_target(df)
    
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)

    # Save scaler instance
    save_scaler(scaler, scaler_path)
    save_scaler(scaler, "scaler.pkl")

    # Baseline Candidate Model Comparison
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=4, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }

    print("[INFO] Evaluating candidate models on water sensor dataset...")
    baseline_scores = {}
    for name, m in models.items():
        if name in ["Decision Tree", "Random Forest"]:
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
        else:
            m.fit(X_train_scaled, y_train)
            pred = m.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred)
        baseline_scores[name] = acc
        print(f"  - {name:<20}: {acc * 100:.2f}% accuracy")

    best_baseline_name = max(baseline_scores, key=baseline_scores.get)
    print(f"\n[INFO] Best Baseline Model: {best_baseline_name} ({baseline_scores[best_baseline_name] * 100:.2f}%)")

    # Hyperparameter Tuning for selected Decision Tree model
    print("[INFO] Performing hyperparameter tuning on Decision Tree Classifier...")
    tuned_model = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=4,
        min_samples_split=10,
        random_state=42
    )
    tuned_model.fit(X_train, y_train)

    # Evaluate metrics on unseen test set (240 samples)
    metrics = evaluate_model_performance(tuned_model, X_test, y_test, is_scaled=False)
    
    print("\n" + "="*50)
    print(f"[TUNED DECISION TREE EVALUATION RESULTS]")
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
            pickle.dump(tuned_model, f)
            
    print(f"\n[SUCCESS] Tuned Decision Tree model saved successfully to '{model_path}' and 'model.pkl'.")
    return tuned_model, scaler, metrics

if __name__ == "__main__":
    train_and_save_model()
