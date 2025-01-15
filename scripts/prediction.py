import os
import joblib
import re

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

def clean_numeric(value):
    """Remove non-numeric characters and convert to float."""
    try:
        return float(re.sub(r'[^\d.]', '', str(value)))
    except ValueError:
        return 0.0  # Default fallback for errors

def encode_categorical(data):
    """Convert categorical features to numerical values."""
    # Encoding mappings for categorical fields
    status_map = {'Ready to Move': 1, 'Under Construction': 0}
    transaction_map = {'New Property': 1, 'Resale': 0}
    furnishing_map = {'Furnished': 2, 'Semi-Furnished': 1, 'Unfurnished': 0}
    facing_map = {'East': 0, 'West': 1, 'North': 2, 'South': 3}
    ownership_map = {'Freehold': 1, 'Leasehold': 0}
    balcony_map = {'Yes': 1, 'No': 0}

    # Apply the mappings
    data['Status'] = status_map.get(data.get('Status'), 0)
    data['Transaction'] = transaction_map.get(data.get('Transaction'), 0)
    data['Furnishing'] = furnishing_map.get(data.get('Furnishing'), 0)
    data['Facing'] = facing_map.get(data.get('Facing'), 0)
    data['Ownership'] = ownership_map.get(data.get('Ownership'), 0)
    data['Balcony'] = balcony_map.get(data.get('Balcony'), 0)

    return data

def predict_price(input_data):
    try:
        # Clean numeric inputs
        input_data['Area'] = clean_numeric(input_data.get('Area', 0))
        input_data['Bathroom'] = clean_numeric(input_data.get('Bathroom', 0))
        input_data['Car Parking'] = clean_numeric(input_data.get('Car Parking', 0))
        input_data['Floor'] = clean_numeric(input_data.get('Floor', 0))

        # Encode categorical inputs
        input_data = encode_categorical(input_data)

        # Define the correct order of features as per the trained model
        features_order = ['Area', 'Status', 'Floor', 'Transaction', 
                          'Furnishing', 'Facing', 'Ownership', 
                          'Car Parking', 'Bathroom', 'Balcony']

        # Prepare input for prediction in the correct order
        input_values = [input_data.get(feature, 0) for feature in features_order]

        # Scale the input data
        input_scaled = scaler.transform([input_values])

        # Predict the price
        prediction = gbr_model.predict(input_scaled)
        return round(prediction[0], 2)  # Return price rounded to 2 decimal places

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return None
