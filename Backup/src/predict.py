# src/predict.py

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load model
model_path = r"D:\marketing-campaign-optimizer\models\random_forest_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found at models/random_forest_model.pkl")

model = joblib.load(model_path)

# Load training data to extract encoders
df_train = pd.read_csv(r"D:\marketing-campaign-optimizer\data\processed_marketing.csv")
X_train = df_train.drop("y", axis=1)

# Categorical features used in training (if needed for encoding consistency)
def load_label_encoders(raw_df):
    """ Re-create label encoders for input encoding """
    categorical = raw_df.select_dtypes(include='object').columns
    encoders = {}
    for col in categorical:
        le = LabelEncoder()
        raw_df[col] = le.fit_transform(raw_df[col])
        encoders[col] = le
    return encoders

# Load raw data to get encoders
df_raw = pd.read_csv(r"D:\marketing-campaign-optimizer\data\bank\bank-full.csv")
encoders = load_label_encoders(df_raw)

def preprocess_input(user_input_dict):
    """ Preprocess a single user input row """
    df_input = pd.DataFrame([user_input_dict])

    # Encode categorical features
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])
    
    return df_input

def predict_campaign_response(user_input_dict):
    """ Predict using the trained model """
    preprocessed = preprocess_input(user_input_dict)
    prediction = model.predict(preprocessed)[0]
    probability = model.predict_proba(preprocessed)[0][1]
    return {
        "prediction": int(prediction),
        "probability_of_yes": round(probability, 3)
    }

# --- Example usage ---
if __name__ == "__main__":
    # Sample input (use same fields as in original dataset)
    sample_input = {
        'age': 35,
        'job': 'admin.',
        'marital': 'married',
        'education': 'secondary',
        'default': 'no',
        'balance': 1000,
        'housing': 'yes',
        'loan': 'no',
        'contact': 'cellular',
        'day': 5,
        'month': 'may',
        'duration': 200,
        'campaign': 1,
        'pdays': -1,
        'previous': 0,
        'poutcome': 'unknown'
    }

    result = predict_campaign_response(sample_input)
    print("Prediction result:", result)
