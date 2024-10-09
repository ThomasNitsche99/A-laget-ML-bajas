# Import necessary libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from geopy.distance import geodesic
import warnings

warnings.filterwarnings('ignore')

# Load the AIS training data
ais_data = pd.read_csv('data/cleaned_data/cleaned_ais_train1.csv')

# Standardize column names
ais_data.columns = ais_data.columns.str.strip().str.upper()

# Data Preprocessing

# Drop rows with missing latitude or longitude
ais_data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

# Convert TIME and ETARAW to datetime objects
ais_data['TIME'] = pd.to_datetime(ais_data['TIME'], errors='coerce')
ais_data['ETARAW'] = pd.to_datetime(ais_data['ETARAW'], errors='coerce')

# Extract time-based features
ais_data['HOUR'] = ais_data['TIME'].dt.hour
ais_data['DAY_OF_WEEK'] = ais_data['TIME'].dt.dayofweek
ais_data['MONTH'] = ais_data['TIME'].dt.month

# Calculate time to ETA in hours
ais_data['TIME_TO_ETA'] = (ais_data['ETARAW'] - ais_data['TIME']).dt.total_seconds() / 3600
ais_data['TIME_TO_ETA'].replace([np.inf, -np.inf], np.nan, inplace=True)
ais_data['TIME_TO_ETA'].fillna(ais_data['TIME_TO_ETA'].median(), inplace=True)

# Filter out invalid HEADING, NAVSTAT values (since these are your core predictors)
ais_data = ais_data[(ais_data['HEADING'] >= 0) & (ais_data['HEADING'] <= 359)]
ais_data = ais_data[ais_data['NAVSTAT'] != 15]  # Assuming 15 is not available

# Defining Target Variables

# Prediction horizon of up to 5 days
prediction_horizon = timedelta(days=5)

# Create future time column
ais_data['FUTURE_TIME'] = ais_data['TIME'] + prediction_horizon

# Prepare future positions
future_positions = ais_data[['VESSELID', 'TIME', 'LATITUDE', 'LONGITUDE']].copy()
future_positions.rename(columns={
    'TIME': 'FUTURE_TIME',
    'LATITUDE': 'FUTURE_LATITUDE',
    'LONGITUDE': 'FUTURE_LONGITUDE'
}, inplace=True)

# Merge future positions back
ais_data = pd.merge(ais_data, future_positions, on=['VESSELID', 'FUTURE_TIME'], how='left')

# Drop rows without future positions
ais_data.dropna(subset=['FUTURE_LATITUDE', 'FUTURE_LONGITUDE'], inplace=True)

# Defining Feature Sets for Latitude and Longitude Predictions

latitude_features = ['HEADING', 'TIME_TO_ETA']
longitude_features = ['TIME_TO_ETA']

# Separate the training data for each prediction task
X_train_lat = ais_data[latitude_features]
X_train_lon = ais_data[longitude_features]

y_train_lat = ais_data['LATITUDE']
y_train_lon = ais_data['LONGITUDE']

# Train RandomForest models
from sklearn.ensemble import RandomForestRegressor

model_lat = RandomForestRegressor(random_state=42)
model_lat.fit(X_train_lat, y_train_lat)

model_lon = RandomForestRegressor(random_state=42)
model_lon.fit(X_train_lon, y_train_lon)

# Model Evaluation

# Define X and y for Latitude and Longitude
X = ais_data[['HEADING', 'TIME_TO_ETA']]
y_lat = ais_data['FUTURE_LATITUDE']
y_lon = ais_data['FUTURE_LONGITUDE']

# Time-based split for train/test sets
ais_data = ais_data.sort_values(by='TIME')
split_index = int(len(ais_data) * 0.8)
X_train = X.iloc[:split_index]
X_valid = X.iloc[split_index:]
y_lat_train = y_lat.iloc[:split_index]
y_lat_valid = y_lat.iloc[split_index:]
y_lon_train = y_lon.iloc[:split_index]
y_lon_valid = y_lon.iloc[split_index:]

# Training the XGBoost Models

# Define parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Model for Latitude
model_lat_xgb = xgb.XGBRegressor(**xgb_params)
model_lat_xgb.fit(X_train[['HEADING', 'TIME_TO_ETA']], y_lat_train)

# Model for Longitude
model_lon_xgb = xgb.XGBRegressor(**xgb_params)
model_lon_xgb.fit(X_train[['TIME_TO_ETA']], y_lon_train)

# Model Evaluation on Validation Set

# Make predictions on validation set
y_lat_pred = model_lat_xgb.predict(X_valid[['HEADING', 'TIME_TO_ETA']])
y_lon_pred = model_lon_xgb.predict(X_valid[['TIME_TO_ETA']])

# Combine predictions and actual values
predictions = pd.DataFrame({
    'LATITUDE_actual': y_lat_valid.values,
    'LONGITUDE_actual': y_lon_valid.values,
    'LATITUDE_pred': y_lat_pred,
    'LONGITUDE_pred': y_lon_pred
})

# Calculate geodesic distance error
def calculate_distance(row):
    actual = (row['LATITUDE_actual'], row['LONGITUDE_actual'])
    pred = (row['LATITUDE_pred'], row['LONGITUDE_pred'])
    return geodesic(actual, pred).kilometers

predictions['error_km'] = predictions.apply(calculate_distance, axis=1)

# Calculate mean error
mean_error = predictions['error_km'].mean()
print(f"Mean Geodesic Distance Error: {mean_error:.2f} km")

# Final Model Performance
xgb.plot_importance(model_lat_xgb, max_num_features=10)
plt.title('Feature Importance for Latitude Prediction')
plt.show()

xgb.plot_importance(model_lon_xgb, max_num_features=10)
plt.title('Feature Importance for Longitude Prediction')
plt.show()

# At this point, you have trained the models and evaluated their performance. 
# You can proceed to load the test dataset for making predictions.



# Load the AIS test dataset
test_data = pd.read_csv('data/ais_test.csv')

# Standardize column names
test_data.columns = test_data.columns.str.strip().str.upper()

# Convert TIME to datetime objects
test_data['TIME'] = pd.to_datetime(test_data['TIME'], errors='coerce')

# Extract necessary time-based features
test_data['TIME_TO_ETA'] = np.nan  # Initialize TIME_TO_ETA in case it's missing

# Handling missing or unavailable values in the test dataset

# Ensure TIME_TO_ETA is available
if 'ETARAW' in test_data.columns:
    test_data['ETARAW'] = pd.to_datetime(test_data['ETARAW'], errors='coerce')
    test_data['TIME_TO_ETA'] = (test_data['ETARAW'] - test_data['TIME']).dt.total_seconds() / 3600
    test_data['TIME_TO_ETA'].replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data['TIME_TO_ETA'].fillna(ais_data['TIME_TO_ETA'].median(), inplace=True)
else:
    # Use median TIME_TO_ETA from training data if not available
    test_data['TIME_TO_ETA'].fillna(ais_data['TIME_TO_ETA'].median(), inplace=True)

# Ensure HEADING is available for latitude prediction
if 'HEADING' not in test_data.columns or test_data['HEADING'].isnull().any():
    print("'HEADING' is missing or contains null values. Filling with median value from training data.")
    test_data['HEADING'] = ais_data['HEADING'].median()  # Fallback to the median of training data


# Prepare the feature set for predictions
latitude_features = ['HEADING', 'TIME_TO_ETA']
longitude_features = ['TIME_TO_ETA']

# Ensure the test dataset has the necessary features
X_test_lat = test_data[latitude_features]
X_test_lon = test_data[longitude_features]

# Predict on test data
test_lat_pred = model_lat.predict(X_test_lat)
test_lon_pred = model_lon.predict(X_test_lon)

# Prepare the output DataFrame
output = pd.DataFrame({
    'ID': test_data['ID'],
    'longitude_predicted': test_lon_pred,
    'latitude_predicted': test_lat_pred
})

# Save the predictions to a CSV file
output.to_csv('submissions/eivind_submissions/submission_3.csv', index=False)

print("Predictions saved to 'submissions/eivind_submissions/submission_3.csv'")