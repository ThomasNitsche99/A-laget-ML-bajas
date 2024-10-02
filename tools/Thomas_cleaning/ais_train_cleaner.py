import pandas as pd

# Load the dataset
file_path = 'data/formatted/ais_train_converted.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

data['original_index'] = data.index


# 1. Remove invalid COG values
# COG should be between 0 and 360. Remove values between 360.1 to 409.5.
data = data[(data['cog'] >= 0) & (data['cog'] <= 360) | (data['cog'].isnull())]

# 2. Remove invalid SOG values
# SOG should be between 0 and 102.2. Remove values above 102.2.
data = data[(data['sog'] >= 0) & (data['sog'] <= 102.2) | (data['sog'].isnull())]

# 3. Remove invalid ROT values
# ROT should be between -126 and +126. Remove -128.
data = data[(data['rot'] >= -127) & (data['rot'] <= 127) | (data['rot'].isnull())]

# 4. Remove invalid Heading values
# Heading should be between 0 and 359. Remove values equal to 511.
data = data[(data['heading'] >= 0) & (data['heading'] <= 359) | (data['heading'].isnull())]

# 5. Handle NAVSTAT codes
# NAVSTAT codes should be between 0 and 15.
data = data[(data['navstat'] >= 0) & (data['navstat'] <= 15) | (data['navstat'].isnull())]

# 6. Clean ETA Raw values
# Ensure ETA Raw is in datetime format or mark as missing.
data['etaRaw'] = pd.to_datetime('2024-' + data['etaRaw'].str.replace(r'(\d{2}-\d{2})', r'\1', regex=True), errors='coerce')

# 7. Validate Latitude and Longitude
# Latitude should be between -90 and 90, Longitude should be between -180 and 180.
data = data[(data['latitude'] >= -90) & (data['latitude'] <= 90) | (data['latitude'].isnull())]
data = data[(data['longitude'] >= -180) & (data['longitude'] <= 180) | (data['longitude'].isnull())]

# 8. Remove duplicate rows
data = data.drop_duplicates()
    
# 9. Handle missing values within the context of VesselID
# Group the data by vesselId and fill missing values within each vessel's group
def fill_missing_values_within_group(group):
    # Replace missing numeric values with the median for each vessel group
    for col in ['cog', 'sog', 'rot', 'latitude', 'longitude', 'heading']:
        group[col] = group[col].fillna(group[col].median())
    return group

# Apply the function to each vesselId group
data = data.groupby('vesselId').apply(fill_missing_values_within_group)

# Drop the group index to restore original structure
data.reset_index(drop=True, inplace=True)

data.sort_values(by='original_index', inplace=True)
data.drop(columns=['original_index'], inplace=True)  # Remove the temporary index column

# Replace missing object values with 'Unknown'
object_columns = ['vesselId', 'portId']
for col in object_columns:
    data[col].fillna('Unknown', inplace=True)


# 10. Final check and save cleaned data
cleaned_file_path = 'cleaned_ais_train_dataset.csv'  # Replace with your desired path
data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}")
