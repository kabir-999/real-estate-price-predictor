import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import re
import os

# Load the dataset
file_path = "data/magicbricks_detailed_properties.csv"
df = pd.read_csv(file_path)

# Clean the 'Area' column
df['Area'] = df['Area'].str.replace(r'[^\d.]', '', regex=True).astype(float)

# Clean the 'Price' column
df['Price'] = df['Price'].str.replace(r'[^\d.]', '', regex=True).astype(float)

# Fill missing numerical values with the median
df['Area'].fillna(df['Area'].median(), inplace=True)

# Fill missing categorical values with mode
for col in ['Status', 'Floor', 'Transaction', 'Furnishing', 'Facing', 'Overlooking', 'Ownership', 'Car Parking', 'Bathroom', 'Balcony']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Extract floor number
def extract_floor(floor):
    numbers = re.findall(r'\d+', str(floor))
    return int(numbers[0]) if numbers else 0

df['Floor'] = df['Floor'].apply(extract_floor)

# Encode categorical features
categorical_cols = ['Status', 'Transaction', 'Furnishing', 'Facing', 'Ownership', 'Balcony']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Clean 'Car Parking' feature
df['Car Parking'] = df['Car Parking'].astype(str).str.extract(r'(\d+)').astype(float)
df['Car Parking'].fillna(0, inplace=True)

# Convert 'Bathroom' to numeric
df['Bathroom'] = df['Bathroom'].replace({'> 10': 11}).astype(int)

# Define features and target (Removed 'Carpet Area')
features = ['Area', 'Status', 'Floor', 'Transaction', 
            'Furnishing', 'Facing', 'Ownership', 'Car Parking', 'Bathroom', 'Balcony']
X = df[features]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

# Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/gbr_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Model and Scaler saved successfully.")
