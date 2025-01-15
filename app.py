from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
model_path = os.path.join('models', 'gbr_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')

gbr_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Mapping dictionaries for categorical features
status_dict = {'Ready to Move': 1, 'Under Construction': 0}
transaction_dict = {'New Property': 1, 'Resale': 0}
furnishing_dict = {'Furnished': 2, 'Semi-Furnished': 1, 'Unfurnished': 0}
facing_dict = {'East': 0, 'West': 1, 'North': 2, 'South': 3}
ownership_dict = {'Freehold': 1, 'Leasehold': 0}
balcony_dict = {'Yes': 1, 'No': 0}

@app.route('/', methods=['GET', 'POST'])
def predict():
    predicted_price = None
    if request.method == 'POST':
        try:
            # Retrieve input values from the form
            area = float(request.form['area'])
            status = status_dict[request.form['status']]
            transaction = transaction_dict[request.form['transaction']]
            furnishing = furnishing_dict[request.form['furnishing']]
            facing = facing_dict[request.form['facing']]
            ownership = ownership_dict[request.form['ownership']]
            balcony = balcony_dict[request.form['balcony']]
            bathroom = int(request.form['bathroom'])
            car_parking = int(request.form['car_parking'])
            floor_number = int(request.form['floor'])

            # Feature array for prediction
            features = np.array([
                area, status, transaction, furnishing, facing,
                ownership, balcony, bathroom, car_parking, floor_number
            ]).reshape(1, -1)

            # Scale features
            scaled_features = scaler.transform(features)

            # Predict price
            predicted_price = round(gbr_model.predict(scaled_features)[0], 2)

        except Exception as e:
            predicted_price = f"Error: {e}"

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
