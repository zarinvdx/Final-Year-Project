import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os

os.makedirs('FYP', exist_ok=True)
df = pd.read_csv('cleaned_vmCloud_data.csv').dropna()
print("CSV loaded")

# Create lag features
df['cpu_lag1'] = df['cpu_usage'].shift(1)
df['cpu_lag2'] = df['cpu_usage'].shift(2)
df['mem_lag1'] = df['memory_usage'].shift(1)
df['mem_lag2'] = df['memory_usage'].shift(2)
df.dropna(inplace=True)

# Select input features used in real-time dashboard prediction
features = [
    'cpu_lag1', 'cpu_lag2',
    'mem_lag1', 'mem_lag2',
    'network_traffic', 'energy_efficiency',
    'hour', 'day_of_week',
    'cpu_usage', 'memory_usage'
]
target = 'power_consumption'

X = df[features]
y = df[target]

print("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest")
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("Evaluating model")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(" Model Evaluation:")
print(f" MSE: {mse:.2f}")
print(f" MAE: {mae:.2f}")
print(f" R² Score: {r2:.4f}")

# Plot feature importance
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.barh([features[i] for i in sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("RandomForest Feature Importance")
plt.tight_layout()
plt.savefig("FYP/feature_importance.png")
print("Feature importance graph saved")

joblib.dump(model, 'FYP/predictor.pkl')
print(" Model saved as predictor.pkl")
