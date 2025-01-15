from flask import Flask, render_template, request
import pandas as pd
from scripts.prediction import predict_price

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_price = None
    if request.method == 'POST':
        try:
            # Collect input data from form
            input_data = pd.DataFrame({
                'Area': [float(request.form['area'])],
                'Status': [request.form['status']],
                'Transaction': [request.form['transaction']],
                'Furnishing': [request.form['furnishing']],
                'Facing': [request.form['facing']],
                'Ownership': [request.form['ownership']],
                'Balcony': [request.form['balcony']],
                'Bathroom': [float(request.form['bathroom'])],
                'Car_Parking': [float(request.form['car_parking'])],
                'Floor': [float(request.form['floor'])]
            })

            # Predict price
            predicted_price = predict_price(input_data)

        except Exception as e:
            print(f"❌ Error processing form data: {e}")
    
    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
