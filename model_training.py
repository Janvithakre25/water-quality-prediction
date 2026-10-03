import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Ensure stdout handles UTF-8 encoding on Windows console if possible
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def train_and_evaluate_models():
    dataset_path = "water_purification_dataset.csv"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found!")

    print("[INFO] Loading Water Purification Dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    # Feature selection
    feature_cols = [
        "pH", "turbidity_NTU", "TDS_ppm", "flow_rate_L_min",
        "pressure_bar", "temperature_C", "usage_L_per_day", "days_since_filter_change"
    ]
    target_col = "water_quality"

    X = df[feature_cols]
    y = df[target_col]

    print("\n[INFO] Missing Values Summary:")
    print(X.isnull().sum())

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_score = 0.0
    best_name = ""

    print("\n[INFO] Evaluating Machine Learning Models...")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f" -> {name}:")
        print(f"   Accuracy: {acc:.4f} | Weighted F1: {f1:.4f} | 5-Fold CV Mean: {cv_scores.mean():.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print("=" * 60)
    print(f"[BEST] Best Base Model: {best_name} (Accuracy: {best_score:.4f})")

    # Hyperparameter Tuning for Best Model
    if "Forest" in best_name or "Tree" in best_name:
        print(f"\n[INFO] Tuning Hyperparameters for {best_name} via GridSearchCV...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        } if "Forest" in best_name else {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(
            estimator=best_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_
        print(f"Optimal Parameters: {grid_search.best_params_}")
        tuned_acc = accuracy_score(y_test, best_model.predict(X_test_scaled))
        print(f"Tuned Test Accuracy: {tuned_acc:.4f}")

    # Final Classification Report
    y_final_pred = best_model.predict(X_test_scaled)
    print("\n[INFO] Final Classification Report:")
    print(classification_report(y_test, y_final_pred, target_names=["Safe (0)", "Moderate (1)", "Unsafe (2)"]))

    print("\n[INFO] Final Confusion Matrix:")
    print(confusion_matrix(y_test, y_final_pred))

    # Save Model & Scaler
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\n[SUCCESS] Saved 'model.pkl' and 'scaler.pkl' successfully!")

if __name__ == "__main__":
    train_and_evaluate_models()
