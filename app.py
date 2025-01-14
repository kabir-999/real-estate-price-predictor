import streamlit as st
import pandas as pd
import joblib
import os

# 📌 Load the trained model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
gbr_model = joblib.load(os.path.join(current_dir, 'models', 'gbr_model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'models', 'scaler.pkl'))

# 📌 Define dropdown options
status_options = ['Ready to Move', 'Under Construction']
transaction_options = ['New Property', 'Resale']
furnishing_options = ['Furnished', 'Semi-Furnished', 'Unfurnished']
facing_options = ['East', 'West', 'North', 'South']
ownership_options = ['Freehold', 'Leasehold']
balcony_options = ['Yes', 'No']
bathroom_options = [1, 2, 3, 4, 5, 6]
car_parking_options = [0, 1, 2, 3, 4]
floor_options = list(range(0, 51))

# 📌 Streamlit Title
st.title("🏠 Real Estate Price Prediction")

# 📌 User Inputs
area = st.number_input("Area (in sqft)", min_value=100, max_value=10000, step=50)
status = st.selectbox("Status", status_options)
transaction = st.selectbox("Transaction Type", transaction_options)
furnishing = st.selectbox("Furnishing Status", furnishing_options)
facing = st.selectbox("Facing Direction", facing_options)
ownership = st.selectbox("Ownership Type", ownership_options)
balcony = st.selectbox("Balcony", balcony_options)
bathroom = st.selectbox("Number of Bathrooms", bathroom_options)
car_parking = st.selectbox("Car Parking Spaces", car_parking_options)
floor = st.selectbox("Floor Number", floor_options)

# 📌 Prediction Button
if st.button("Predict Price"):
    # Prepare input data
    user_data = {
        'Area': area,
        'Status': status,
        'Transaction': transaction,
        'Furnishing': furnishing,
        'Facing': facing,
        'Ownership': ownership,
        'Balcony': balcony,
        'Bathroom': bathroom,
        'Car Parking': car_parking,
        'Floor': floor,
        'Price_per_sqft': area / (area + 1)
    }

    input_df = pd.DataFrame([user_data])

    # Encode categorical data
    label_encodable_cols = ['Status', 'Transaction', 'Furnishing', 'Facing', 'Ownership', 'Balcony']
    for col in label_encodable_cols:
        input_df[col] = pd.factorize(input_df[col])[0]

    # Align input with model features
    model_features = ['Area', 'Status', 'Transaction', 'Furnishing', 'Facing',
                      'Ownership', 'Balcony', 'Bathroom', 'Car Parking', 'Floor', 'Price_per_sqft']
    input_df = input_df[model_features]

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Predict price
    predicted_price = gbr_model.predict(input_scaled)
    st.success(f"🏠 Predicted Property Price: ₹{round(predicted_price[0], 2)} Lakhs")
