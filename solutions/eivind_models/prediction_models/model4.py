# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU
import datetime

# Load the AIS dataset (train)
def load_data():
    # Adjust path to the actual dataset path
    train_data = pd.read_csv('data/cleaned_data/cleaned_ais_train1.csv')
    test_data = pd.read_csv('data/ais_test.csv')
   
    return train_data, test_data

# Preprocess the AIS data
def preprocess_data(train_data, test_data):
    # Feature Engineering for Time
    # Convert 'time' column to datetime
    train_data['time'] = pd.to_datetime(train_data['time'])
    test_data['time'] = pd.to_datetime(test_data['time'])
    
    # Sort by vessel and time for time-series handling
    train_data = train_data.sort_values(by=['vesselId', 'time'])
    
    # Create time features (e.g., time difference between consecutive records)
    train_data['time_diff'] = train_data.groupby('vesselId')['time'].diff().dt.total_seconds().fillna(0)
    
    # Extract time-based features (hour of the day, day of the week, etc.)
    train_data['hour'] = train_data['time'].dt.hour
    train_data['day_of_week'] = train_data['time'].dt.dayofweek
    
    # Handle categorical data like vesselId
    le = LabelEncoder()
    train_data['vesselId'] = le.fit_transform(train_data['vesselId'])
    
    # Scaling features
    scaler = StandardScaler()
    features_to_scale = ['SOG', 'COG', 'ROT', 'Heading', 'time_diff', 'hour', 'day_of_week']
    train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
    
    # Define the target (latitude, longitude)
    target_columns = ['latitude', 'longitude']
    
    # Drop irrelevant columns or those not available during test time
    train_data.drop(columns=['portId', 'ETARAW'], inplace=True)
    
    return train_data, test_data, scaler, le

# Create LSTM/GRU model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(2))  # Output is (latitude, longitude)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare time-series data for LSTM/GRU
def prepare_time_series_data(train_data, target_columns, time_steps=5):
    # Define the features and target for training
    features = train_data.drop(columns=target_columns)
    targets = train_data[target_columns]
    
    X, y = [], []
    for vessel_id in train_data['vesselId'].unique():
        vessel_data = train_data[train_data['vesselId'] == vessel_id]
        
        # Create time windows (time_steps) for LSTM
        for i in range(time_steps, len(vessel_data)):
            X.append(vessel_data.iloc[i-time_steps:i].values)
            y.append(targets.iloc[i].values)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Train the model
def train_model(X, y):
    input_shape = (X.shape[1], X.shape[2])  # time_steps, number of features
    model = create_lstm_model(input_shape)
    model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2)
    return model

# Predict vessel positions for test data
def predict_positions(model, test_data, scaler, le):
    # Preprocess test data (apply same scaling, encoding as train)
    test_data['vesselId'] = le.transform(test_data['vesselId'])
    test_data['time_diff'] = test_data.groupby('vesselId')['time'].diff().dt.total_seconds().fillna(0)
    test_data['hour'] = test_data['time'].dt.hour
    test_data['day_of_week'] = test_data['time'].dt.dayofweek
    test_data[['time_diff', 'hour', 'day_of_week']] = scaler.transform(test_data[['time_diff', 'hour', 'day_of_week']])
    
    # Prepare test time-series data similar to training data
    predictions = []
    ids = []  # To store IDs corresponding to predictions
    
    for idx, vessel_id in enumerate(test_data['vesselId'].unique()):
        vessel_data = test_data[test_data['vesselId'] == vessel_id]
        
        if len(vessel_data) < 5:  # Check if there's enough data for prediction
            print(f"Not enough data for vesselId: {vessel_id}. Skipping.")
            continue
        
        time_window = vessel_data.iloc[-5:].values  # Use last available data as window
        predicted_position = model.predict(time_window[np.newaxis, :, :])
        
        # Save the ID and predicted positions
        for i in range(5):  # Assuming predicting for the next 5 time steps
            ids.append(len(predictions))  # ID for predictions
            predictions.append(predicted_position[0])  # Append predicted latitude/longitude
            
    return ids, predictions

# Save predictions to CSV
def save_predictions(ids, predictions):
    # Convert predictions to a DataFrame
    pred_df = pd.DataFrame(predictions, columns=['longitude_predicted', 'latitude_predicted'])
    pred_df['ID'] = ids
    
    # Reorder columns as specified
    pred_df = pred_df[['ID', 'longitude_predicted', 'latitude_predicted']]
    
    # Save to CSV
    pred_df.to_csv('submissions/eivind_submissions/submission4.csv', index=False)
    print("Predictions saved to predictions.csv")

# Main workflow
def main():
    # Load data
    train_data, test_data = load_data()
    
    # Preprocess data
    train_data, test_data, scaler, le = preprocess_data(train_data, test_data)
    
    # Prepare time-series data for training
    target_columns = ['latitude', 'longitude']
    X, y = prepare_time_series_data(train_data, target_columns, time_steps=5)
    
    # Train the model
    model = train_model(X, y)
    
    # Predict on test data
    ids, predictions = predict_positions(model, test_data, scaler, le)
    
    # Save predictions to CSV
    save_predictions(ids, predictions)

if __name__ == '__main__':
    main()
