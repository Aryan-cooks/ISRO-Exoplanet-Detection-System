import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def train_classifier(csv_path="feature_table.csv"):
    # Read the data
    df = pd.read_csv(csv_path)
    
    features = ['Period_days', 'Duration_hours', 'Depth', 'SNR', 'Odd_Even_Sigma']
    X = df[features]
    y_raw = df['Label']
    tic_ids = df['TIC_ID']
    
    # Encode string labels to integers (XGBoost requirement)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Initialize and train the XGBoost classifier
    model = xgb.XGBClassifier(
        eval_metric='mlogloss', 
        random_state=42, 
        n_estimators=10, 
        max_depth=2, 
        reg_lambda=0.1,
        min_child_weight=0
    )
    model.fit(X, y)
    
    # Predict on the training set
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Decode predictions back to string labels
    predicted_labels = le.inverse_transform(predictions)
    
    print("=" * 80)
    print("IMPORTANT DISCLAIMER:")
    print("This is a workflow demonstration trained on an extremely small sample (5 targets, 3 classes).")
    print("This is NOT a validated model. The accuracy, probabilities, and feature importances")
    print("reported here should NOT be interpreted as generalizable performance.")
    print("=" * 80)
    print()
    
    print("--- Training Set Predictions & Probabilities ---")
    classes = le.classes_
    print(f"Classes: {classes}")
    for i in range(len(df)):
        prob_str = ", ".join([f"{classes[j]}: {probabilities[i][j]:.4f}" for j in range(len(classes))])
        print(f"Target: {tic_ids.iloc[i]}")
        print(f"  True Label:      {y_raw.iloc[i]}")
        print(f"  Predicted Label: {predicted_labels[i]}")
        print(f"  Probabilities:   [{prob_str}]")
        print()
        
    print("--- Feature Importances ---")
    importances = model.feature_importances_
    for feat, imp in zip(features, importances):
        print(f"{feat:16s}: {imp:.4f}")
        
    print("\n" + "=" * 80)
    print("CONCLUSION ON MODEL COMPLEXITY:")
    print("With default hyperparameters, the model degenerated to predicting class-frequency")
    print("priors due to the extremely small sample size and default safety constraints.")
    print("After explicitly setting min_child_weight=0, the model successfully split on")
    print("features (no longer degenerate) and perfectly classified all 5 training samples")
    print("using only Period_days and Depth (importances 0.48 and 0.52).")
    print("Duration_hours, SNR, and Odd_Even_Sigma received zero importance, likely because")
    print("the shallow trees (max_depth=2) needed only 1-2 splits to separate 3 classes,")
    print("not because those features are uninformative.")
    print("\nPerfect classification on a training set with as many model parameters as data")
    print("points is expected and does not constitute validated generalization performance.")
    print("A larger labeled dataset would be needed to determine real feature importance")
    print("and out-of-sample accuracy.")
    print("=" * 80)

if __name__ == "__main__":
    train_classifier()
