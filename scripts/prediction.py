import os
import joblib
import pandas as pd

# Load the model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'models', 'gbr_model.pkl')
scaler_path = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')
dataset_path = os.path.join(BASE_DIR, '..', 'data', 'magicbricks_detailed_properties.csv')  # Add your dataset

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
    print(f"❌ Error loading dataset: {e}")

def predict_price(input_data):
    try:
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        # Predict price
        prediction = gbr_model.predict(input_scaled)
        return prediction[0]

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        # Fallback: Predict closest price from dataset
        return fallback_prediction(input_data)

def fallback_prediction(input_data):
    """
    Find the closest matching property in the dataset and return its price.
    """
    try:
        # Calculate the difference between input data and dataset properties
        diff = dataset.drop(columns=['Price']).apply(lambda row: ((row - input_data.iloc[0]) ** 2).sum(), axis=1)
        closest_index = diff.idxmin()
        closest_price = dataset.iloc[closest_index]['Price']
        print(f"🔍 Closest Match Found. Price: {closest_price}")
        return closest_price

    except Exception as e:
        print(f"❌ Fallback Prediction Error: {e}")
        return "Error occurred. Please try again."
