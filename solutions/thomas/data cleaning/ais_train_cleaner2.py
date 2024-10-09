import pandas as pd
import numpy as np

# Load the ais_train_dataset
file_path = 'data/formatted/ais_train_converted.csv' 
train_data = pd.read_csv(file_path)

#original index for restoring structure
train_data['original_index'] = train_data.index

#Convert time to datetime
train_data['time'] = pd.to_datetime(train_data['time'])

# Handle invalid COG values
train_data.loc[train_data['cog'] == 360, 'cog'] = np.nan
train_data = train_data[(train_data['cog'].isnull()) | ((train_data['cog'] >= 0) & (train_data['cog'] <= 359))]

# Handle invalid SOG values
train_data.loc[train_data['sog'] == 1023, 'sog'] = np.nan
train_data.loc[train_data['sog'] > 102.2, 'sog'] = 102.2

# Handle invalid ROT values
train_data.loc[train_data['rot'] == -128, 'rot'] = np.nan
train_data = train_data[(train_data['rot'].isnull()) | ((train_data['rot'] >= -126) & (train_data['rot'] <= 126))]

# Handle invalid Heading values
train_data.loc[train_data['heading'] == 511, 'heading'] = np.nan
train_data = train_data[(train_data['heading'].isnull()) | ((train_data['heading'] >= 0) & (train_data['heading'] <= 359))]

# Handle NAVSTAT codes
train_data = train_data[(train_data['navstat'].isnull()) | ((train_data['navstat'] >= 0) & (train_data['navstat'] <= 15))]

# Clean ETA Raw values
# train_data['etaRaw'] = pd.to_datetime(train_data['etaRaw'], errors='coerce', utc=True)
train_data['etaRaw'] = pd.to_datetime('2024-' + train_data['etaRaw'].str.replace(r'(\d{2}-\d{2})', r'\1', regex=True), errors='coerce')


# Validate Latitude and Longitude
train_data = train_data[(train_data['latitude'].isnull()) | ((train_data['latitude'] >= -90) & (train_data['latitude'] <= 90))]
train_data = train_data[(train_data['longitude'].isnull()) | ((train_data['longitude'] >= -180) & (train_data['longitude'] <= 180))]

# Remove duplicate rows
train_data = train_data.drop_duplicates()

# Handle missing values within the context of VesselID
def fill_missing_values_within_group(group):
    for col in ['cog', 'sog', 'rot', 'latitude', 'longitude', 'heading']:
        group[col] = group[col].fillna(group[col].median())
    return group

train_data = train_data.groupby('vesselId', group_keys=False).apply(fill_missing_values_within_group)

# Drop the group index to restore original structure
train_data.reset_index(drop=True, inplace=True)

# Sort and clean up
train_data.sort_values(by='original_index', inplace=True)
train_data.drop(columns=['original_index'], inplace=True)

# Replace missing object values with 'Unknown'
object_columns = ['vesselId', 'portId']
for col in object_columns:
    train_data[col].fillna('Unknown', inplace=True)
    
# 5. Dropping columns not needed
train_data.drop('etaRaw', axis=1, inplace=True)
train_data.drop('portId', axis=1, inplace=True) #Remove for now, not using port dataset

train_data = train_data.sort_values(by=['vesselId', 'time'])

# Final check and save cleaned train_data
cleaned_file_path = 'data/cleaned/clean2/cleaned2_ais_train_dataset.csv'  # Replace with your desired path
train_data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned train_dataset saved to {cleaned_file_path}")
