from flask import Flask, render_template, request
from scripts.prediction import predict_price
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    if request.method == 'POST':
        try:
            # Collect input data
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
            predicted_price = round(predicted_price, 2)

        except Exception as e:
            print(f"❌ Error: {e}")
            predicted_price = "Error occurred. Please try again."

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
