import streamlit as st
import pandas as pd
import joblib
import os

# 📌 Dynamically get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# 📌 Load the trained model and scaler
gbr_model = joblib.load(os.path.join(current_dir, 'models', 'gbr_model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'models', 'scaler.pkl'))

# 📌 Streamlit App Title
st.title("🏠 Real Estate Price Prediction App")

# 📌 Input Fields
area = st.number_input("Enter Area (in sqft)", min_value=100, max_value=10000, step=50)
status = st.selectbox("Status", ['Ready to Move', 'Under Construction'])
transaction = st.selectbox("Transaction Type", ['New Property', 'Resale'])
furnishing = st.selectbox("Furnishing Status", ['Furnished', 'Semi-Furnished', 'Unfurnished'])
facing = st.selectbox("Facing Direction", ['East', 'West', 'North', 'South'])
ownership = st.selectbox("Ownership Type", ['Freehold', 'Leasehold'])
balcony = st.selectbox("Balcony", ['Yes', 'No'])
bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
car_parking = st.number_input("Car Parking Spaces", min_value=0, max_value=5, step=1)
floor = st.number_input("Floor Number", min_value=0, max_value=50, step=1)

# 📌 Predict Button
if st.button("Predict Price"):
    # Prepare the input data
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

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # Encode categorical features
    for col in ['Status', 'Transaction', 'Furnishing', 'Facing', 'Ownership', 'Balcony']:
        input_df[col] = input_df[col].astype('category').cat.codes

    # Align features with the model
    model_features = ['Area', 'Status', 'Transaction', 'Furnishing', 'Facing', 'Ownership',
                      'Balcony', 'Bathroom', 'Car Parking', 'Floor', 'Price_per_sqft']

    input_df = input_df[model_features]

    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Predict
    predicted_price = gbr_model.predict(input_scaled)
    st.success(f"🏠 Predicted Property Price: ₹{round(predicted_price[0], 2)} Lakhs")
