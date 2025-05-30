import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
csv_path = 'dataset.csv'
df = pd.read_csv(csv_path)

# Select relevant features (inputs)
feature_columns = [
    'angle', 'rpm', 'speedX', 'speedY', 'speedZ', 'trackPos'
] + [f'track_{i}' for i in range(19)]
# Optionally add opponent sensors and wheel spin if you want:
# feature_columns += [f'opponent_{i}' for i in range(36)]
# feature_columns += [f'wheelSpinVel_{i}' for i in range(4)]

# Select targets (outputs)
target_columns = ['accel', 'brake', 'steer']

# Drop rows with missing values (if any)
df = df.dropna(subset=feature_columns + target_columns)

X = df[feature_columns].astype(float)
y = df[target_columns].astype(float)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu', max_iter=100, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluate
y_pred = mlp.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse:.4f}')

# Save model and scaler
joblib.dump(mlp, 'torcs_mlp_model.joblib')
joblib.dump(scaler, 'torcs_scaler.joblib')

print('Model and scaler saved!')

# ---
# Why MLPRegressor?
# - Handles nonlinear relationships between sensors and controls.
# - Fast to train, easy to tune, and works well for tabular data.
# - You can later switch to PyTorch or TensorFlow for more advanced models or if you want to deploy on GPU. 