import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def train_models(data_path="water_purification_dataset.csv"):
    print("=" * 50)
    print("Smart Water Quality Prediction - Model Training")
    print("=" * 50)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at path: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Define features and target
    feature_cols = [
        "pH", "turbidity_NTU", "TDS_ppm", "flow_rate_L_min",
        "pressure_bar", "temperature_C", "usage_L_per_day", "days_since_filter_change"
    ]
    target_col = "water_quality"
    
    X = df[feature_cols]
    y = df[target_col]
    
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
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }
    
    best_model = None
    best_score = 0.0
    best_name = ""
    
    print("\nEvaluating Candidate Models:")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        print(f"[{name}]")
        print(f"  Test Accuracy : {acc * 100:.2f}%")
        print(f"  5-Fold CV Mean: {cv_scores.mean() * 100:.2f}% (std: {cv_scores.std() * 100:.2f}%)")
        
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    # Hyperparameter Tuning on Decision Tree
    if best_name == "Decision Tree":
        print("\nTuning Decision Tree Hyperparameters...")
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        print(f"Best Parameters: {grid.best_params_}")
        best_score = grid.best_score_

    print("-" * 50)
    print(f"Selected Model: {best_name} (Accuracy: {best_score * 100:.2f}%)")
    
    # Evaluate final best model on test set
    y_final_pred = best_model.predict(X_test_scaled)
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_final_pred, target_names=["Safe (0)", "Moderate (1)", "Unsafe (2)"]))
    
    # Save model and scaler
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("[SUCCESS] Model saved to model.pkl")
    print("[SUCCESS] Scaler saved to scaler.pkl")
    print("=" * 50)

if __name__ == "__main__":
    train_models()
