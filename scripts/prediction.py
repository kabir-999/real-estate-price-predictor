import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# 📌 Load the Trained Model and Scaler
gbr_model = joblib.load('./models/gbr_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

# 📌 User Input
def user_input_prediction():
    area = float(input("Enter Area in sqft (e.g., 2500): "))
    status = input("Choose Status (Ready to Move / Under Construction): ")
    transaction = input("Choose Transaction Type (New Property / Resale): ")
    furnishing = input("Choose Furnishing Status (Furnished / Semi-Furnished / Unfurnished): ")
    facing = input("Choose Facing Direction (East / West / North / South): ")
    ownership = input("Choose Ownership Type (Freehold / Leasehold): ")
    balcony = input("Does it have a Balcony? (Yes/No): ")
    bathroom = int(input("Enter Number of Bathrooms: "))
    car_parking = int(input("Enter Number of Car Parking Spaces: "))
    floor = int(input("Enter Floor Number: "))

    # 📌 Prepare Data
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
    label_encodable_cols = ['Status', 'Transaction', 'Furnishing', 'Facing', 'Ownership', 'Balcony']
    
    label_encoder = LabelEncoder()
    for col in label_encodable_cols:
        input_df[col] = label_encoder.fit_transform(input_df[col].astype(str))

    # 📌 Align and Scale
    input_scaled = scaler.transform(input_df)
    
    # 📌 Prediction
    predicted_price = gbr_model.predict(input_scaled)
    print(f"\n🏠 Predicted Property Price: ₹{round(predicted_price[0], 2)} Lakhs\n")

user_input_prediction()
