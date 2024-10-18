import pandas as pd
import numpy as np

# Step 1: Load the AIS dataset
df = pd.read_csv('/Users/eivindmidtbo/Desktop/dev/Maskinlaering_i_praksis/A-laget-ML-bajas/data/ais_train.csv', sep='|')

# Step 2: Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'])

# Step 3: Define the Haversine function to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in nautical miles (1 nautical mile = 1.15078 miles)
    r = 3440.065  # in nautical miles
    return c * r

# Step 4: Initialize a list to keep track of the indices to drop
indices_to_drop = []

# Step 5: Set parameters for filtering
speed_threshold = 30  # Speed threshold in knots
max_distance = 50  # Maximum distance threshold in nautical miles (adjust as needed)

# Step 6: Process each vessel's data
for vessel_id in df['vesselId'].unique():
    vessel_data = df[df['vesselId'] == vessel_id]
    vessel_data = vessel_data.sort_values('time')  # Ensure chronological order
    
    # Step 7: Calculate distances and time differences
    distances = []
    time_diffs = []
    
    for i in range(1, len(vessel_data)):
        # Calculate distance
        dist = haversine(vessel_data['latitude'].iloc[i-1], vessel_data['longitude'].iloc[i-1],
                         vessel_data['latitude'].iloc[i], vessel_data['longitude'].iloc[i])
        
        # Calculate time difference in hours
        time_diff = (vessel_data['time'].iloc[i] - vessel_data['time'].iloc[i-1]).total_seconds() / 3600  # Convert to hours
        
        # Append distance and time difference
        distances.append(dist)
        time_diffs.append(time_diff)
        
        # Step 8: Calculate speed and check against thresholds
        if time_diff > 0:  # Avoid division by zero
            speed = dist / time_diff  # Speed in nautical miles per hour
            # Check if speed or distance exceeds thresholds
            if speed > speed_threshold or dist > max_distance:
                indices_to_drop.append(vessel_data.index[i])  # Mark this index for removal

# Step 9: Drop the indices from the DataFrame
df_cleaned = df.drop(index=indices_to_drop)

# Optional: Reset the index of the cleaned DataFrame
df_cleaned.reset_index(drop=True, inplace=True)

# Step 10: Save the cleaned dataset to a new CSV file (optional)
df_cleaned.to_csv('/Users/eivindmidtbo/Desktop/dev/Maskinlaering_i_praksis/A-laget-ML-bajas/data/cleaned_data/ais_train_cleaned2.csv', sep='|', index=False)

print(f"Cleaned dataset saved. Total records before cleaning: {len(df)}, after cleaning: {len(df_cleaned)}.")
