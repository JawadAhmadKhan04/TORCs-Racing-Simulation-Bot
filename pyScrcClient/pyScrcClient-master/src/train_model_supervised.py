import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

# === 1. Load and clean dataset ===
df = pd.read_csv("dataset.csv")

# Define input features
input_features = [
    "speedX", "speedY", "speedZ", "angle", "trackPos"
] + [f"track_{i}" for i in range(19)] + [f"opponent_{i}" for i in range(36)]

# Define output labels
output_labels = ["accel", "brake", "steer", "gear"]

# Drop rows with missing values
df = df[input_features + output_labels].dropna()

X = df[input_features].values
y = df[output_labels].values

# === 2. Normalize inputs ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 4. Train 4 separate XGBoost regressors ===
models = {}
for i, label in enumerate(output_labels):
    print(f"Training model for: {label}")
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train[:, i])
    models[label] = model

# === 5. Evaluate and save models ===
for label in output_labels:
    score = models[label].score(X_test, y_test[:, output_labels.index(label)])
    print(f"{label} R² score: {score:.4f}")
    joblib.dump(models[label], f"xgb_model_{label}.pkl")

# Save the scaler
joblib.dump(scaler, "xgb_scaler.pkl")

print("✅ All models and scaler saved.")
