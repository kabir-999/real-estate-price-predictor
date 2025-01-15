from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'gbr_model.pkl')
scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')

gbr_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Encoding dictionaries for categorical inputs
status_dict = {'Ready to Move': 1, 'Under Construction': 0}
transaction_dict = {'New Property': 1, 'Resale': 0}
furnishing_dict = {'Furnished': 2, 'Semi-Furnished': 1, 'Unfurnished': 0}
facing_dict = {'East': 0, 'West': 1, 'North': 2, 'South': 3}
ownership_dict = {'Freehold': 1, 'Leasehold': 0}
balcony_dict = {'Yes': 1, 'No': 0}

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None

    if request.method == "POST":
        try:
            # Collect input data
            area = float(request.form["area"])
            status = request.form["status"]
            transaction = request.form["transaction"]
            furnishing = request.form["furnishing"]
            facing = request.form["facing"]
            ownership = request.form["ownership"]
            balcony = request.form["balcony"]
            bathroom = int(request.form["bathroom"])
            car_parking = int(request.form["car_parking"])
            floor = int(request.form["floor"])

            # Preprocess input
            input_features = np.array([
                area,
                status_dict[status],
                transaction_dict[transaction],
                furnishing_dict[furnishing],
                facing_dict[facing],
                ownership_dict[ownership],
                balcony_dict[balcony],
                bathroom,
                car_parking,
                floor,
                area / (area + 1)
            ]).reshape(1, -1)

            # Scale the features
            scaled_features = scaler.transform(input_features)

            # Predict the price
            predicted_price = round(gbr_model.predict(scaled_features)[0], 2)

        except Exception as e:
            predicted_price = f"Error: {e}"

    return render_template("index.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
