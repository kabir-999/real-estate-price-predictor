# 📌 Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 📌 Load Dataset
df = pd.read_csv('./data/magicbricks_detailed_properties.csv')
df.drop(columns=['Location'], inplace=True)

# 📌 Clean and Process Data
df['Price'] = df['Price'].str.replace(r'[^\d.]', '', regex=True).astype(float)
df['Area'] = df['Area'].str.replace(r'[^\d.]', '', regex=True).astype(float)
df.fillna(df.median(), inplace=True)

# 📌 Encode Categorical Features
label_encodable_cols = ['Status', 'Transaction', 'Furnishing', 'Facing', 'Ownership', 'Balcony']
label_encoder = LabelEncoder()
for col in label_encodable_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# 📌 Feature Engineering
df['Price_per_sqft'] = df['Price'] / (df['Area'] + 1)
X = df.drop(columns=['Title', 'Price', 'Carpet Area', 'Overlooking'])
y = df['Price']

# 📌 Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 📌 Initialize and Train Model
gbr_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
gbr_model.fit(X_train, y_train)

# 📌 Save the Model and Scaler
joblib.dump(gbr_model, './models/gbr_model.pkl')
joblib.dump(scaler, './models/scaler.pkl')

print("✅ Model and Scaler saved successfully!")
