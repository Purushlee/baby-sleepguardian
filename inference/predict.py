import joblib
import numpy as np
from features.extract_features import extract_features

# Load model & scaler
model = joblib.load('../models/binary_model.pkl')
scaler = joblib.load('../models/scaler_binary.pkl')

def predict(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])
    prediction = model.predict(features)
    return prediction[0]

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    result = predict(file_path)
    print(f"Prediction: {result}")
