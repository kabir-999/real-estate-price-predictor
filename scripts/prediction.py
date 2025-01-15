import os
import joblib

# Correct path to the model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'gbr_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Load the model and scaler
gbr_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predict_price(input_data):
    # Assuming input_data is a DataFrame
    input_scaled = scaler.transform(input_data)
    prediction = gbr_model.predict(input_scaled)
    return prediction[0]
