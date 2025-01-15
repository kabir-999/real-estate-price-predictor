import os
import joblib
import sys
import types
import numpy as np
import pandas as pd

# -------------------------
# FIX for missing '_loss' module
# -------------------------
# Create a dummy '_loss' module if it's missing
sys.modules['_loss'] = types.ModuleType('_loss')

# -------------------------
# Load the model and scaler
# -------------------------

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the trained model and scaler
model_path = os.path.join(BASE_DIR, '..', 'models', 'gbr_model.pkl')
scaler_path = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

# Load the model and scaler
try:
    gbr_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# -------------------------
# Prediction Function
# -------------------------
def predict_price(input_data):
    """
    Predicts property price based on input data.

    :param input_data: DataFrame with input features
    :return: Predicted price
    """
    try:
        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Predict the price
        prediction = gbr_model.predict(input_scaled)

        return prediction[0]  # Return the predicted price
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None
