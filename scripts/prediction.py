from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# 📌 Load Model and Scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'gbr_model.pkl')
scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 📌 Dictionaries for Encoding
status_dict = {'Ready to Move': 1, 'Under Construction': 0}
transaction_dict = {'New Property': 1, 'Resale': 0}
furnishing_dict = {'Furnished': 2, 'Semi-Furnished': 1, 'Unfurnished': 0}
facing_dict = {'East': 0, 'West': 1, 'North': 2, 'South': 3}
ownership_dict = {'Freehold': 1, 'Leasehold': 0}
balcony_dict = {'Yes': 1, 'No': 0}

# 📌 Home Page
@app.route('/')
def home():
    return render_template('index.html')

# 📌 Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        status = request.form['status']
        transaction = request.form['transaction']
        furnishing = request.form['furnishing']
        facing = request.form['facing']
        ownership = request.form['ownership']
        balcony = request.form['balcony']
        bathroom = int(request.form['bathroom'])
        car_parking = int(request.form['car_parking'])

        # 📌 Encode Input
        input_data = np.array([
            area,
            status_dict[status],
            transaction_dict[transaction],
            furnishing_dict[furnishing],
            facing_dict[facing],
            ownership_dict[ownership],
            balcony_dict[balcony],
            bathroom,
            car_parking,
            area / (area + 1)  # Derived feature
        ]).reshape(1, -1)

        # 📌 Scale Input
        scaled_input = scaler.transform(input_data)

        # 📌 Predict
        prediction = model.predict(scaled_input)[0]
        return render_template('result.html', price=round(prediction, 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
