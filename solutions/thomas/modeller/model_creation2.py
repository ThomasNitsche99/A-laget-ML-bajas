# Import necessary libraries
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import timedelta
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import math
from tqdm import tqdm

# ----------------------------
# 1. Load and Preprocess Training Data
# ----------------------------

# Load the cleaned ais_train dataset
train_file_path = 'data/cleaned/cleaned2_ais_train_dataset.csv'
train_data = pd.read_csv(train_file_path, parse_dates=['time'])

# Sort the data by vesselId and time
train_data = train_data.sort_values(by=['vesselId', 'time'])

# ----------------------------
# 2. Feature Engineering on Training Data
# ----------------------------

def feature_engineering_train(df):
    df = df.sort_values(by=['vesselId', 'time'])
    
    # Time Difference in seconds
    df['time_diff'] = df.groupby('vesselId')['time'].diff().dt.total_seconds()
    
    # Previous Positions
    df['prev_latitude'] = df.groupby('vesselId')['latitude'].shift(1)
    df['prev_longitude'] = df.groupby('vesselId')['longitude'].shift(1)
    
    # Change in Position
    df['delta_latitude'] = df['latitude'] - df['prev_latitude']
    df['delta_longitude'] = df['longitude'] - df['prev_longitude']
    
    # Distance Traveled
    def calculate_distance(row):
        if pd.notnull(row['prev_latitude']) and pd.notnull(row['prev_longitude']):
            coords_1 = (row['prev_latitude'], row['prev_longitude'])
            coords_2 = (row['latitude'], row['longitude'])
            return geodesic(coords_1, coords_2).kilometers
        else:
            return np.nan
    df['distance_traveled'] = df.apply(calculate_distance, axis=1)
    
    # Movement Differences
    df['sog_diff'] = df.groupby('vesselId')['sog'].diff()
    df['acceleration'] = df['sog_diff'] / df['time_diff']
    df['cog_diff'] = df.groupby('vesselId')['cog'].diff()
    df['rot_diff'] = df.groupby('vesselId')['rot'].diff()
    df['drift'] = df['heading'] - df['cog']
    
    # Time-based Features
    df['hour'] = df['time'].dt.hour + df['time'].dt.minute / 60
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # One-Hot Encode navstat
    navstat_dummies = pd.get_dummies(df['navstat'], prefix='navstat')
    df = pd.concat([df, navstat_dummies], axis=1)
    
    # Drop rows with NaN values resulting from feature engineering
    df = df.dropna()
    
    return df, navstat_dummies.columns.tolist()

print("Running feature engineering on training data...")
train_data_fe, navstat_columns = feature_engineering_train(train_data)
print("Feature engineering on training data complete.")

# ----------------------------
# 3. Prepare Features and Labels for Modeling
# ----------------------------

# Define feature columns
feature_columns = [
    'sog', 'cog', 'rot', 'heading',
    'sog_diff', 'acceleration', 'cog_diff', 'rot_diff', 'drift',
    'distance_traveled', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
] + navstat_columns

# Extract features (X) and targets (y)
X = train_data_fe[feature_columns]
y = train_data_fe[['latitude', 'longitude']]

# ----------------------------
# 4. Split Data into Training and Validation Sets
# ----------------------------

# Since it's time-series data, split based on time to prevent data leakage
train_data_fe['timestamp'] = train_data_fe['time'].values.astype(np.int64) // 10 ** 9
timestamp_threshold = train_data_fe['timestamp'].quantile(0.8)

X_train = X[train_data_fe['timestamp'] <= timestamp_threshold]
X_val = X[train_data_fe['timestamp'] > timestamp_threshold]
y_train = y[train_data_fe['timestamp'] <= timestamp_threshold]
y_val = y[train_data_fe['timestamp'] > timestamp_threshold]

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")

# ----------------------------
# 5. Define and Evaluate Models Using Cross-Validation
# ----------------------------

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, n_jobs=8, random_state=42, max_depth=4),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror', max_depth=3, learning_rate=0.01), 
    'LightGBM' : lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
}

# Function to evaluate models using cross-validation
def evaluate_models_cv(X, y, models):
    results = {}
    for name, model in tqdm(models.items(), desc="Evaluating models"):
        print(f"Evaluating model: {name}")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(model))
        ])
        kf = KFold(n_splits=3, shuffle=False)  # No shuffling for time-series
        # Use negative MSE as scoring
        cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        mean_cv_mse = -np.mean(cv_scores)
        results[name] = mean_cv_mse
        print(f"{name}: Mean CV MSE = {mean_cv_mse}\n")
    return results

print("Evaluating models using cross-validation...")
results = evaluate_models_cv(X_train, y_train, models)

# Select the best model (lowest MSE)
best_model_name = min(results, key=results.get)
best_mse = results[best_model_name]
print(f"Best model: {best_model_name} with Mean CV MSE = {best_mse}")

# ----------------------------
# 6. Retrain the Best Model on the Entire Training Set
# ----------------------------

# Initialize the best model
best_model = models[best_model_name]

# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', MultiOutputRegressor(best_model))
])

# Fit the pipeline on the entire training data
print("Retraining the best model on the entire training set...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# ----------------------------
# 7. Prepare and Feature Engineering for Test Data
# ----------------------------

# Load the test dataset
test_file_path = 'data/raw/ais_test.csv'  # Replace with your actual test dataset path
test_data = pd.read_csv(test_file_path, parse_dates=['time'])

print("Loading and preparing test data...")

# Since the test data lacks AIS features, we need to generate them using historical AIS data from train_data_fe

# Merge test data with the latest AIS data before the test time for each vessel
def prepare_test_features(test_df, train_df, feature_columns):
    # Sort both dataframes
    test_df = test_df.sort_values(by=['vesselId', 'time'])
    train_df = train_df.sort_values(by=['vesselId', 'time'])
    
    # Initialize lists to store features
    feature_list = []
    
    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Processing Test Data"):
        vessel_id = row['vesselId']
        test_time = row['time']
        
        # Get historical data for the vessel up to the test_time
        vessel_history = train_df[(train_df['vesselId'] == vessel_id) & (train_df['time'] < test_time)]
        
        if vessel_history.empty:
            # If no historical data, fill features with median or default values
            features = [0] * len(feature_columns)
        else:
            # Use the last available AIS record before test_time
            last_record = vessel_history.iloc[-1]
            
            # Create a single-row DataFrame for feature engineering
            temp_df = pd.DataFrame([last_record])
            
            # Feature Engineering on the last record
            # Note: Only one record, so differences will be NaN and thus need to be filled
            temp_df['time_diff'] = np.nan
            temp_df['prev_latitude'] = vessel_history['latitude'].iloc[-2] if len(vessel_history) >= 2 else np.nan
            temp_df['prev_longitude'] = vessel_history['longitude'].iloc[-2] if len(vessel_history) >= 2 else np.nan
            temp_df['delta_latitude'] = temp_df['latitude'] - temp_df['prev_latitude'] if pd.notnull(temp_df['prev_latitude']).all() else 0
            temp_df['delta_longitude'] = temp_df['longitude'] - temp_df['prev_longitude'] if pd.notnull(temp_df['prev_longitude']).all() else 0
            
            # Distance traveled
            if pd.notnull(temp_df['prev_latitude']).all() and pd.notnull(temp_df['prev_longitude']).all():
                coords_1 = (temp_df['prev_latitude'].values[0], temp_df['prev_longitude'].values[0])
                coords_2 = (temp_df['latitude'].values[0], temp_df['longitude'].values[0])
                temp_df['distance_traveled'] = geodesic(coords_1, coords_2).kilometers
            else:
                temp_df['distance_traveled'] = 0.0
            
            # Movement Differences
            temp_df['sog_diff'] = 0.0  # Assuming no change
            temp_df['acceleration'] = 0.0  # Assuming no acceleration
            temp_df['cog_diff'] = 0.0  # Assuming no change in course
            temp_df['rot_diff'] = 0.0  # Assuming no change in ROT
            temp_df['drift'] = temp_df['heading'] - temp_df['cog']
            
            # Time-based Features
            temp_df['hour'] = temp_df['time'].dt.hour + temp_df['time'].dt.minute / 60
            temp_df['hour_sin'] = np.sin(2 * np.pi * temp_df['hour'] / 24)
            temp_df['hour_cos'] = np.cos(2 * np.pi * temp_df['hour'] / 24)
            temp_df['day_of_week'] = temp_df['time'].dt.dayofweek
            temp_df['day_sin'] = np.sin(2 * np.pi * temp_df['day_of_week'] / 7)
            temp_df['day_cos'] = np.cos(2 * np.pi * temp_df['day_of_week'] / 7)
            
            # One-Hot Encode navstat
            navstat_dummies_test = pd.get_dummies(temp_df['navstat'], prefix='navstat')
            # Ensure all navstat columns from training are present
            for col in navstat_columns:
                if col not in navstat_dummies_test.columns:
                    navstat_dummies_test[col] = 0
            temp_df = pd.concat([temp_df, navstat_dummies_test], axis=1)
            
            # Fill NaN values if any
            temp_df.fillna(0, inplace=True)
            
            # Select feature columns
            features = temp_df[feature_columns].values.flatten()
        
        feature_list.append(features)
    
    # Create a DataFrame for test features
    X_test = pd.DataFrame(feature_list, columns=feature_columns)
    return X_test

print("Generating features for test data...")
X_test = prepare_test_features(test_data, train_data_fe, feature_columns)
print("Feature generation for test data complete.")

# ----------------------------
# 8. Handle Missing Features in Test Data
# ----------------------------

# Ensure that all feature columns are present in the test data
missing_cols = set(feature_columns) - set(X_test.columns)
for col in tqdm(missing_cols):
    X_test[col] = 0  # Fill missing columns with default values (e.g., 0)

# Reorder columns to match training data
X_test = X_test[feature_columns]

# ----------------------------
# 9. Make Predictions on Test Data
# ----------------------------

print("Making predictions on test data...")
predictions = pipeline.predict(X_test)
print("Predictions complete.")

# ----------------------------
# 10. Create and Save the Submission CSV
# ----------------------------

# Load the original test_data to get the 'ID' column
test_ids = test_data['ID']

# Create the submission DataFrame
submission = pd.DataFrame({
    'ID': test_ids,
    'longitude_predicted': predictions[:, 1],
    'latitude_predicted': predictions[:, 0]
})

# Save the predictions to a CSV file
submission_file_path = 'predictions/submission.csv'
submission.to_csv(submission_file_path, index=False)
print(f"Predictions saved to {submission_file_path}")
