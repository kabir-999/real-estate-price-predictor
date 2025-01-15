from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'gbr_model.pkl')
scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')

try:
    gbr_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        status = int(request.form['status'])
        transaction = int(request.form['transaction'])
        furnishing = int(request.form['furnishing'])
        facing = int(request.form['facing'])
        ownership = int(request.form['ownership'])
        balcony = int(request.form['balcony'])
        bathroom = int(request.form['bathroom'])
        car_parking = int(request.form['car_parking'])
        floor = int(request.form['floor'])
        
        input_features = np.array([[area, status, transaction, furnishing, facing, ownership, balcony, bathroom, car_parking, floor]])
        scaled_features = scaler.transform(input_features)
        prediction = gbr_model.predict(scaled_features)
        
        return render_template('index.html', prediction_text=f'Predicted Property Price: ₹{round(prediction[0], 2)} Crores')
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
