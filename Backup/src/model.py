# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# Load preprocessed data
df = pd.read_csv(r"D:\marketing-campaign-optimizer\data\processed_marketing.csv")

# Features and target
X = df.drop("y", axis=1)
y = df["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Logistic Regression ---
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("\n--- Logistic Regression Evaluation ---")
print(confusion_matrix(y_test, lr_preds))
print(classification_report(y_test, lr_preds))
print("ROC AUC:", roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1]))

# --- Random Forest Classifier ---
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n--- Random Forest Evaluation ---")
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
print("ROC AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))

# Save best model (e.g., Random Forest)
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, r"D:\marketing-campaign-optimizer\models\random_forest_model.pkl")
print("\n✅ Model saved to models/random_forest_model.pkl")
