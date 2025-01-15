from flask import Flask, render_template, request
from scripts.prediction import predict_price

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    error_message = None

    if request.method == 'POST':
        input_data = {
            'Area': request.form['Area'],
            'Status': request.form['Status'],
            'Transaction': request.form['Transaction'],
            'Furnishing': request.form['Furnishing'],
            'Facing': request.form['Facing'],
            'Ownership': request.form['Ownership'],
            'Balcony': request.form['Balcony'],
            'Bathroom': request.form['Bathroom'],
            'Car Parking': request.form['Car Parking'],
            'Floor': request.form['Floor']
        }

        predicted_price, error_message = predict_price(input_data)

    return render_template('index.html', predicted_price=predicted_price, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
