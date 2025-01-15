import joblib
import numpy as np
import os

# Load the model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'gbr_model.pkl')
scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')

gbr_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Encoding dictionaries
status_dict = {'Ready to Move': 1, 'Under Construction': 0}
transaction_dict = {'New Property': 1, 'Resale': 0}
furnishing_dict = {'Furnished': 2, 'Semi-Furnished': 1, 'Unfurnished': 0}
facing_dict = {'East': 0, 'West': 1, 'North': 2, 'South': 3}
ownership_dict = {'Freehold': 1, 'Leasehold': 0}
balcony_dict = {'Yes': 1, 'No': 0}

def preprocess_input(area, status, transaction, furnishing, facing, ownership, balcony, bathroom, car_parking, floor):
    features = np.array([
        area,
        status_dict[status],
        transaction_dict[transaction],
        furnishing_dict[furnishing],
        facing_dict[facing],
        ownership_dict[ownership],
        balcony_dict[balcony],
        bathroom,
        car_parking,
        floor
    ]).reshape(1, -1)

    scaled_features = scaler.transform(features)
    return scaled_features

def predict_price(area, status, transaction, furnishing, facing, ownership, balcony, bathroom, car_parking, floor):
    processed_input = preprocess_input(area, status, transaction, furnishing, facing, ownership, balcony, bathroom, car_parking, floor)
    predicted_price = gbr_model.predict(processed_input)
    return round(predicted_price[0], 2)
