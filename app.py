from flask import Flask, render_template, request
from scripts.prediction import predict_price
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    error_message = None

    if request.method == 'POST':
        try:
            # Collect input data from form
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

            # Prepare data for prediction
            input_data = pd.DataFrame([{
                'Area': area,
                'Status': status,
                'Transaction': transaction,
                'Furnishing': furnishing,
                'Facing': facing,
                'Ownership': ownership,
                'Balcony': balcony,
                'Bathroom': bathroom,
                'Car_Parking': car_parking,
                'Floor': floor
            }])

            # Predict price
            predicted_price = predict_price(input_data)

            if predicted_price is not None:
                predicted_price = round(predicted_price, 2)
            else:
                error_message = "Prediction failed. Please try again."

        except Exception as e:
            error_message = f"Error occurred: {e}"

    return render_template('index.html', predicted_price=predicted_price, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
