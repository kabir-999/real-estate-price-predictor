import os
import joblib

# Define the correct path to the model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'models', 'gbr_model.pkl')
scaler_path = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

# Load the model and scaler
try:
    gbr_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")

def predict_price(input_data):
    try:
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = gbr_model.predict(input_scaled)
        return round(prediction[0], 2)
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return None
