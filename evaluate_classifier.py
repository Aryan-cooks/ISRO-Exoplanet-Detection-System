import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def evaluate():
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    try:
        test_df = pd.read_csv("test_data.csv")
        model = joblib.load("model.pkl")
        le = joblib.load("label_encoder.pkl")
    except Exception as e:
        print(f"Error loading files: {e}")
        return
        
    y_true = test_df['Label']
    X_test = test_df.drop('Label', axis=1)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved metrics.json: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    # Classification Report
    target_names = le.classes_
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\nClassification Report:\n", report)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Plot Feature Importance
    # Extract the classifier from the ImbPipeline
    clf = model.named_steps['clf']
    
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        features = X_test.columns
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in indices]
        sorted_importances = importances[indices]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), sorted_importances, align="center")
        plt.yticks(range(len(indices)), sorted_features)
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300)
        plt.close()
        print("Saved plots/confusion_matrix.png and plots/feature_importance.png")
    else:
        print("Classifier does not have feature_importances_ attribute.")

if __name__ == "__main__":
    evaluate()
