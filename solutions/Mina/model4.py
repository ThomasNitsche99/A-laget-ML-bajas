# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from geopy.distance import geodesic, great_circle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the AIS training data
ais_data = pd.read_csv('data/cleaned/cleaned_ais_train_dataset.csv')

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

# Add columns for the last known position
ais_data['LAST_LATITUDE'] = ais_data.groupby('VESSELID')['LATITUDE'].shift(1)
ais_data['LAST_LONGITUDE'] = ais_data.groupby('VESSELID')['LONGITUDE'].shift(1)

# Fill missing values for the first observation of each vessel
ais_data['LAST_LATITUDE'].fillna(ais_data['LATITUDE'], inplace=True)
ais_data['LAST_LONGITUDE'].fillna(ais_data['LONGITUDE'], inplace=True)

# Calculate distance from the last position
ais_data['DISTANCE_FROM_LAST'] = ais_data.apply(
    lambda row: geodesic((row['LAST_LATITUDE'], row['LAST_LONGITUDE']), 
                         (row['LATITUDE'], row['LONGITUDE'])).kilometers, axis=1)

# Drop unnecessary columns
ais_data.drop(['ETARAW', 'PREV_SOG', 'NAVSTAT', 'ROT_CATEGORY'], axis=1, inplace=True)

# Scaling numeric features
scaler = StandardScaler()
numeric_features = ['COG', 'SOG', 'HEADING', 'RELATIVE_BEARING', 'ACCELERATION', 'DISTANCE_FROM_LAST']
ais_data[numeric_features] = scaler.fit_transform(ais_data[numeric_features])

# Defining Target Variables
prediction_horizon = timedelta(days=5)
ais_data['FUTURE_TIME'] = ais_data['TIME'] + prediction_horizon
future_positions = ais_data[['VESSELID', 'TIME', 'LATITUDE', 'LONGITUDE']].copy()
future_positions.rename(columns={'TIME': 'FUTURE_TIME', 'LATITUDE': 'FUTURE_LATITUDE', 'LONGITUDE': 'FUTURE_LONGITUDE'}, inplace=True)
ais_data = pd.merge(ais_data, future_positions, on=['VESSELID', 'FUTURE_TIME'], how='left')
ais_data.dropna(subset=['FUTURE_LATITUDE', 'FUTURE_LONGITUDE'], inplace=True)

# Select features
feature_cols = [
    'COG', 'SOG', 'HEADING', 'RELATIVE_BEARING', 'ACCELERATION',
    'NAVSTATUS_ENCODED', 'ROT_CATEGORY_ENCODED', 'HOUR', 'DAY_OF_WEEK', 'MONTH',
    'TIME_TO_ETA', 'LAST_LATITUDE', 'LAST_LONGITUDE', 'DISTANCE_FROM_LAST'
]

# Define X and y
X = ais_data[feature_cols]
y_lat = ais_data['FUTURE_LATITUDE']
y_lon = ais_data['FUTURE_LONGITUDE']

# Continue with training, validation, and testing as before

