import os
import joblib
import pandas as pd
import numpy as np

# Define the correct path to the model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'models', 'gbr_model.pkl')
scaler_path = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')
dataset_path = os.path.join(BASE_DIR, '..', 'data', 'magicbricks_detailed_properties.csv')  # Assuming your dataset

# Load the model and scaler
try:
    gbr_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")

# Load dataset for fallback prediction
try:
    dataset = pd.read_csv(dataset_path)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    dataset = None
    print(f"❌ Error loading dataset: {e}")

def predict_price(input_data):
    try:
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = gbr_model.predict(input_scaled)
        return prediction[0]
    
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        # Fallback to the closest prediction if an error occurs
        return fallback_prediction(input_data)

def fallback_prediction(input_data):
    if dataset is None:
        return "Fallback failed: Dataset not available."

    try:
        # Compute similarity by comparing input data with dataset
        dataset_features = dataset.drop('Price', axis=1)  # Assuming 'Price' is the target
        dataset_scaled = scaler.transform(dataset_features)

        input_scaled = scaler.transform(input_data)
        distances = np.linalg.norm(dataset_scaled - input_scaled, axis=1)
        
        # Find the closest record
        closest_index = np.argmin(distances)
        closest_price = dataset.iloc[closest_index]['Price']
        
        print(f"🔍 Fallback Prediction: Closest match price is ₹{closest_price} Crores")
        return closest_price

    except Exception as e:
        print(f"❌ Fallback Prediction Error: {e}")
        return "Error occurred during fallback prediction."
