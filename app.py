from flask import Flask, render_template, request
import pandas as pd
from scripts.prediction import predict_price

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_price = None

    if request.method == 'POST':
        try:
            # Collect user input
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

            # Prepare the data in DataFrame format
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

            # Predict the price
            predicted_price = predict_price(input_data)
        
        except Exception as e:
            print(f"❌ Error in app.py: {e}")
            predicted_price = "Prediction failed. Please try again."

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
