import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ✅ Load the preprocessed dataset
file_path = "D:\\Visual Code\\fetal 6.0\\Data_Preprocess\\new_dataset.csv"
data = pd.read_csv(file_path)

# ✅ Ensure correct target column name
target_col = "BWt(kg)"  # Update if needed
if target_col not in data.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

# ✅ Split Features and Target
X = data.drop(columns=[target_col])
y = data[target_col]

# ✅ One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# ✅ Save column order for Flask input processing
joblib.dump(X.columns.tolist(), "model/columns.pkl")

# ✅ Standardize numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "model/scaler.pkl")  # Save scaler

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# ✅ Save both models
joblib.dump(rf_model, "model/rf_model.pkl")  # Save Random Forest model
joblib.dump(lr_model, "model/lr_model.pkl")  # Save Linear Regression model

print("✅ Random Forest and Linear Regression models, scaler, and columns saved successfully!")
