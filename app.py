from flask import Flask, render_template, request
from scripts.prediction import predict_price

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_price = None
    if request.method == 'POST':
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

        # Prepare input data as a DataFrame
        import pandas as pd
        input_data = pd.DataFrame([{
            'Area': area,
            'Status': status,
            'Transaction': transaction,
            'Furnishing': furnishing,
            'Facing': facing,
            'Ownership': ownership,
            'Balcony': balcony,
            'Bathroom': bathroom,
            'Car Parking': car_parking,
            'Floor': floor
        }])

        # Get the prediction
        predicted_price = predict_price(input_data)

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
