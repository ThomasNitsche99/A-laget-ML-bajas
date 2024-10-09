# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from geopy.distance import geodesic
import matplotlib.pyplot as plt
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

# Filter out invalid COG, SOG, ROT, HEADING values
ais_data = ais_data[(ais_data['COG'] >= 0) & (ais_data['COG'] <= 359)]
ais_data = ais_data[(ais_data['SOG'] >= 0) & (ais_data['SOG'] <= 102.2)]
ais_data = ais_data[ais_data['ROT'] != -128]
ais_data = ais_data[(ais_data['HEADING'] >= 0) & (ais_data['HEADING'] <= 359)]

# Handle NAVSTAT codes
ais_data['NAVSTAT'] = ais_data['NAVSTAT'].astype(int)
ais_data = ais_data[ais_data['NAVSTAT'] != 15]  # Assuming 15 indicates not available

# Feature Engineering

# Calculate relative bearing
ais_data['RELATIVE_BEARING'] = ais_data['HEADING'] - ais_data['COG']
ais_data['RELATIVE_BEARING'] = (ais_data['RELATIVE_BEARING'] + 180) % 360 - 180

# Categorize ROT into bins
rot_bins = [-127, -5, -0.5, 0.5, 5, 127]
rot_labels = ['Hard Port Turn', 'Port Turn', 'Straight', 'Starboard Turn', 'Hard Starboard Turn']
ais_data['ROT_CATEGORY'] = pd.cut(ais_data['ROT'], bins=rot_bins, labels=rot_labels)

# Encode ROT_CATEGORY
ais_data['ROT_CATEGORY_ENCODED'] = LabelEncoder().fit_transform(ais_data['ROT_CATEGORY'].astype(str))

# Calculate acceleration
ais_data = ais_data.sort_values(by=['VESSELID', 'TIME'])
ais_data['PREV_SOG'] = ais_data.groupby('VESSELID')['SOG'].shift(1)
ais_data['ACCELERATION'] = ais_data['SOG'] - ais_data['PREV_SOG']
ais_data['ACCELERATION'].fillna(0, inplace=True)

# Encode NAVSTAT
ais_data['NAVSTATUS_ENCODED'] = LabelEncoder().fit_transform(ais_data['NAVSTAT'])

# Drop unnecessary columns
ais_data.drop(['ETARAW', 'PREV_SOG', 'NAVSTAT', 'ROT_CATEGORY'], axis=1, inplace=True)

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

# Preparing Features and Target

# Select features
feature_cols = [
    'COG', 'SOG', 'ROT', 'HEADING', 'RELATIVE_BEARING', 'ACCELERATION',
    'NAVSTATUS_ENCODED', 'ROT_CATEGORY_ENCODED', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'TIME_TO_ETA'
]

# Define X and y for Latitude and Longitude
X = ais_data[feature_cols]
y_lat = ais_data['FUTURE_LATITUDE']
y_lon = ais_data['FUTURE_LONGITUDE']

# Splitting the Data

# Time-based split
ais_data = ais_data.sort_values(by='TIME')
split_index = int(len(ais_data) * 0.8)
X_train = X.iloc[:split_index]
X_valid = X.iloc[split_index:]
y_lat_train = y_lat.iloc[:split_index]
y_lat_valid = y_lat.iloc[split_index:]
y_lon_train = y_lon.iloc[:split_index]
y_lon_valid = y_lon.iloc[split_index:]

# Training the XGBoost Regression Models

# Define parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Model for Latitude
model_lat = xgb.XGBRegressor(**xgb_params)
model_lat.fit(X_train, y_lat_train)

# Model for Longitude
model_lon = xgb.XGBRegressor(**xgb_params)
model_lon.fit(X_train, y_lon_train)

# Model Evaluation

# Make predictions on validation set
y_lat_pred = model_lat.predict(X_valid)
y_lon_pred = model_lon.predict(X_valid)

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

# Weighted Error Calculation

# Calculate time difference in days
ais_valid = ais_data.iloc[split_index:].copy()
ais_valid['time_diff_days'] = (ais_valid['FUTURE_TIME'] - ais_valid['TIME']).dt.total_seconds() / (24 * 3600)

# Merge time differences
predictions['time_diff_days'] = ais_valid['time_diff_days'].values

# Assign weights
def assign_weight(days):
    if 0 <= days <= 1:
        return 0.3
    elif 1 < days <= 2:
        return 0.25
    elif 2 < days <= 3:
        return 0.2
    elif 3 < days <= 4:
        return 0.15
    elif 4 < days <= 5:
        return 0.1
    else:
        return 0

predictions['weight'] = predictions['time_diff_days'].apply(assign_weight)

# Calculate weighted mean error
weighted_mean_error = np.average(predictions['error_km'], weights=predictions['weight'])
print(f"Weighted Mean Geodesic Distance Error: {weighted_mean_error:.2f} km")

# Feature Importance Analysis
xgb.plot_importance(model_lat, max_num_features=10)
plt.title('Feature Importance for Latitude Prediction')
plt.show()

xgb.plot_importance(model_lon, max_num_features=10)
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

# Extract time-based features
test_data['HOUR'] = test_data['TIME'].dt.hour
test_data['DAY_OF_WEEK'] = test_data['TIME'].dt.dayofweek
test_data['MONTH'] = test_data['TIME'].dt.month

# Handling missing or unavailable values in the test dataset
# Check if ETARAW is in test_data (if it's available)
if 'ETARAW' in test_data.columns:
    test_data['ETARAW'] = pd.to_datetime(test_data['ETARAW'], errors='coerce')
    test_data['TIME_TO_ETA'] = (test_data['ETARAW'] - test_data['TIME']).dt.total_seconds() / 3600
    test_data['TIME_TO_ETA'].replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data['TIME_TO_ETA'].fillna(ais_data['TIME_TO_ETA'].median(), inplace=True)
else:
    # Use median TIME_TO_ETA from training data if not available
    test_data['TIME_TO_ETA'] = ais_data['TIME_TO_ETA'].median()

# Handle missing values in numeric features
numeric_features = ['COG', 'SOG', 'ROT', 'HEADING']
for feature in numeric_features:
    if feature in test_data.columns:
        test_data[feature].fillna(ais_data[feature].median(), inplace=True)
    else:
        # If the feature is missing, fill with median from training data
        test_data[feature] = ais_data[feature].median()

# Handle NAVSTAT
if 'NAVSTAT' in test_data.columns:
    test_data['NAVSTAT'] = test_data['NAVSTAT'].astype(int)
    test_data['NAVSTATUS_ENCODED'] = LabelEncoder().fit_transform(test_data['NAVSTAT'])
else:
    # Use the most common NAVSTATUS_ENCODED from training data
    test_data['NAVSTATUS_ENCODED'] = ais_data['NAVSTATUS_ENCODED'].mode()[0]

# Calculate relative bearing
if 'HEADING' in test_data.columns and 'COG' in test_data.columns:
    test_data['RELATIVE_BEARING'] = test_data['HEADING'] - test_data['COG']
    test_data['RELATIVE_BEARING'] = (test_data['RELATIVE_BEARING'] + 180) % 360 - 180

# Categorize ROT into bins
if 'ROT' in test_data.columns:
    test_data['ROT_CATEGORY'] = pd.cut(
        test_data['ROT'],
        bins=rot_bins,
        labels=rot_labels
    )
    # Encode ROT_CATEGORY
    test_data['ROT_CATEGORY_ENCODED'] = LabelEncoder().fit_transform(test_data['ROT_CATEGORY'].astype(str))
else:
    test_data['ROT_CATEGORY_ENCODED'] = 0  # Set to 0 if ROT is not available

# Calculate acceleration
# Since we might not have previous SOG, set acceleration to 0
test_data['ACCELERATION'] = 0

# Prepare test features
X_test = test_data[feature_cols]

# Ensure that the order of columns matches
X_test = X_test[feature_cols]

# Predict on test data
test_lat_pred = model_lat.predict(X_test)
test_lon_pred = model_lon.predict(X_test)

# Prepare the output DataFrame
output = pd.DataFrame({
    'ID': test_data['ID'],
    'longitude_predicted': test_lon_pred,
    'latitude_predicted': test_lat_pred
})

# Save the predictions to a CSV file
output.to_csv('submissions/eivind_submissions/submission_1.csv', index=False)

print("Predictions saved to 'submissions/eivind_submissions/submission_1.csv'")
