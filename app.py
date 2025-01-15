from flask import Flask, render_template, request
import joblib
import os
from prediction import predict_price

app = Flask(__name__)

# Load the model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'gbr_model.pkl')
scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')

gbr_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_price = None
    if request.method == 'POST':
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
            floor = int(request.form['floor'])

            predicted_price = predict_price(
                area, status, transaction, furnishing,
                facing, ownership, balcony, bathroom, car_parking, floor
            )

        except Exception as e:
            predicted_price = f"Error: {e}"

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
