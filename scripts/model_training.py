import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# 📌 Load dataset
df = pd.read_csv('data/magicbricks_detailed_properties.csv')

# 📌 Drop unnecessary columns
df.drop(columns=['Location'], inplace=True)

# 📌 Clean data
df['Price'] = df['Price'].str.replace(r'[^\d.]', '', regex=True).astype(float)
df['Area'] = df['Area'].str.replace(r'[^\d.]', '', regex=True).astype(float)

# 📌 Encode categorical features
label_encodable_cols = ['Status', 'Transaction', 'Furnishing', 'Facing', 'Ownership', 'Balcony']
encoder = LabelEncoder()
for col in label_encodable_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# 📌 Feature engineering
df['Price_per_sqft'] = df['Price'] / (df['Area'] + 1)
df.fillna(df.median(), inplace=True)

# 📌 Split data
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 Train model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# 📌 Save model and scaler
joblib.dump(model, 'models/gbr_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Model training complete and saved!")
