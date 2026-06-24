import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
import json
import os

warnings.filterwarnings('ignore')

REQUIRED_FEATURES = [
    'Period_days', 'Duration_hours', 'Depth', 'SNR', 'Odd_Even_Sigma',
    'blend_probability', 'neighbor_count', 'bls_peak_power', 
    'transit_symmetry_score', 'v_shape_score', 'u_shape_score',
    'ingress_duration', 'egress_duration', 'num_observed_transits'
]

def load_and_validate(csv_path="curated_dataset.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return None, None, None, None, None
        
    dataset_type = "real"
    if 'TIC_ID' in df.columns and df['TIC_ID'].astype(str).str.contains('MOCK').any():
        dataset_type = "synthetic"
        print("WARNING: Synthetic dataset detected. Using mock data for training.")
        
    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        print(f"WARNING: Missing required features: {missing}. Filling with 0 to prevent crash, but this will degrade performance.")
        for col in missing:
            df[col] = 0.0
            
    if 'Label' not in df.columns:
        print("Error: 'Label' column missing. Cannot train without labels.")
        return None, None, None, None, None
        
    X = df[REQUIRED_FEATURES]
    y_raw = df['Label']
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Save test set for later evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Save the test set and label encoder to disk
    test_df = X_test.copy()
    test_df['Label'] = y_test
    test_df.to_csv("test_data.csv", index=False)
    joblib.dump(le, "label_encoder.pkl")
    
    return X_train, X_test, y_train, y_test, dataset_type

def train():
    X_train, X_test, y_train, y_test, dataset_type = load_and_validate()
    if X_train is None:
        return
        
    print(f"Training set size: {len(X_train)}")
    
    # Define pipelines
    rf_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    xgb_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', xgb.XGBClassifier(eval_metric='mlogloss', random_state=42))
    ])
    
    # Define hyperparameter grids
    rf_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_leaf': [1, 2, 4]
    }
    
    xgb_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.1, 0.2]
    }
    
    print("Optimizing Random Forest...")
    rf_search = RandomizedSearchCV(rf_pipeline, rf_param_grid, n_iter=5, cv=5, scoring='f1_macro', random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    print(f"Best RF Score: {rf_search.best_score_:.4f}")
    
    print("Optimizing XGBoost...")
    xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_param_grid, n_iter=5, cv=5, scoring='f1_macro', random_state=42, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    print(f"Best XGB Score: {xgb_search.best_score_:.4f}")
    
    # Compare and save best
    if xgb_search.best_score_ > rf_search.best_score_:
        best_model = xgb_search.best_estimator_
        print("XGBoost performed best. Saving model.")
    else:
        best_model = rf_search.best_estimator_
        print("Random Forest performed best. Saving model.")
        
    # Update metrics.json
    metrics = {}
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                pass
                
    metrics['dataset_type'] = dataset_type
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    joblib.dump(best_model, "model.pkl")
    print("Model saved to model.pkl")
    print(f"Metrics updated in metrics.json with dataset_type='{dataset_type}'")

if __name__ == "__main__":
    train()
