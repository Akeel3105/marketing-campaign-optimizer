# src/model.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Construct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RAW_PATH = os.path.join(DATA_DIR, 'bank', 'bank-full.csv')
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed_marketing.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'random_forest_model.pkl')

# Load raw or processed data
df = pd.read_csv(PROCESSED_PATH)

# Train/test split
X = df.drop("y", axis=1)
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, rf.predict(X_test)))

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(rf, MODEL_PATH)
