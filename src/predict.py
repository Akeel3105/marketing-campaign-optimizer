# src/predict.py

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

RAW_PATH = os.path.join(DATA_DIR, 'bank', 'bank-full.csv')
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed_marketing.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')

# Load model
model = joblib.load(MODEL_PATH)

# Load data for encoders
df_train = pd.read_csv(PROCESSED_PATH)
X_train = df_train.drop("y", axis=1)

df_raw = pd.read_csv(RAW_PATH)

def load_label_encoders(raw_df):
    encoders = {}
    categorical = raw_df.select_dtypes(include='object').columns
    for col in categorical:
        le = LabelEncoder()
        raw_df[col] = le.fit_transform(raw_df[col])
        encoders[col] = le
    return encoders

encoders = load_label_encoders(df_raw)

def preprocess_input(user_input_dict):
    df_input = pd.DataFrame([user_input_dict])
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])
    return df_input

def predict_campaign_response(user_input_dict):
    preprocessed = preprocess_input(user_input_dict)
    prediction = model.predict(preprocessed)[0]
    probability = model.predict_proba(preprocessed)[0][1]
    return {
        "prediction": int(prediction),
        "probability_of_yes": round(probability, 3)
    }

if __name__ == "__main__":
    sample = {
        'age': 35, 'job': 'admin.', 'marital': 'married', 'education': 'secondary',
        'default': 'no', 'balance': 1000, 'housing': 'yes', 'loan': 'no',
        'contact': 'cellular', 'day': 5, 'month': 'may', 'duration': 200,
        'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'
    }

    print(predict_campaign_response(sample))
