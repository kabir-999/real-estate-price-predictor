import os
import joblib
import pandas as pd
import numpy as np

# Load model and scaler paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'models', 'gbr_model.pkl')
scaler_path = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')
dataset_path = os.path.join(BASE_DIR, '..', 'data', 'magicbricks_detailed_properties.csv')  # Optional dataset

# Load model and scaler
try:
    gbr_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")

# Load dataset for fallback
try:
    dataset = pd.read_csv(dataset_path)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    dataset = None
    print(f"❌ Error loading dataset: {e}")

def predict_price(input_data):
    try:
        print(f"🔎 Input data for prediction:\n{input_data}")
        
        # Ensure feature alignment
        required_features = ['Area', 'Status', 'Transaction', 'Furnishing', 'Facing', 
                             'Ownership', 'Balcony', 'Bathroom', 'Car_Parking', 'Floor']
        input_data = input_data[required_features]

        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Predict price
        prediction = gbr_model.predict(input_scaled)
        return round(prediction[0], 2)

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return fallback_prediction(input_data)

def fallback_prediction(input_data):
    if dataset is None:
        return "Error: Dataset not available for fallback."

    try:
        dataset_features = dataset.drop('Price', axis=1)
        dataset_scaled = scaler.transform(dataset_features)
        
        input_scaled = scaler.transform(input_data)
        distances = np.linalg.norm(dataset_scaled - input_scaled, axis=1)
        
        closest_index = np.argmin(distances)
        closest_price = dataset.iloc[closest_index]['Price']
        
        print(f"🔍 Fallback prediction: ₹{closest_price} Crores")
        return round(closest_price, 2)

    except Exception as e:
        print(f"❌ Fallback Error: {e}")
        return "Error occurred during fallback prediction."
