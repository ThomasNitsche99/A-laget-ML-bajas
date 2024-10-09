import pandas as pd
import numpy as np

# Load the dataset
file_path = "data/formatted/schedules_to_may_2024_converted.csv"  # Replace with the path to your dataset
schedule_data = pd.read_csv(file_path)

#removing duplicates before processing (There are 87 158 duplicates in the datasets)
schedule_data.drop_duplicates(inplace=True)

# Temporary index to preserve original order
schedule_data['original_index'] = schedule_data.index

# 1. Parse 'arrivalDate' and 'sailingDate' into datetime objects (Removed UTC = True)
schedule_data['arrivalDate'] = pd.to_datetime(schedule_data['arrivalDate'], errors='coerce', utc=False)
schedule_data['sailingDate'] = pd.to_datetime(schedule_data['sailingDate'], errors='coerce', utc=False)

# 2. Ensure at least one of 'arrivalDate' or 'sailingDate' is present
schedule_data = schedule_data[
    schedule_data['arrivalDate'].notnull() | schedule_data['sailingDate'].notnull()
]

# 3. Validate 'portLatitude' and 'portLongitude' values
schedule_data = schedule_data[
    ((schedule_data['portLatitude'] >= -90) & (schedule_data['portLatitude'] <= 90)) | schedule_data['portLatitude'].isnull()
]
schedule_data = schedule_data[
    ((schedule_data['portLongitude'] >= -180) & (schedule_data['portLongitude'] <= 180)) | schedule_data['portLongitude'].isnull()
]

# 4. Handle missing values in categorical columns
categorical_columns = ['vesselId', 'shippingLineId', 'shippingLineName', 'portName', 'portId']
for col in categorical_columns:
    schedule_data[col].fillna('Unknown', inplace=True)

# 5. Remove duplicates
schedule_data.drop_duplicates(inplace=True)

# 6. Restore the original order and drop the temporary index column
schedule_data.sort_values(by='original_index', inplace=True)
schedule_data.drop(columns=['original_index'], inplace=True)

# Load the cleaned ports dataset
ports_data = pd.read_csv("data/cleaned/clean2/cleaned2_ports_dataset.csv")

# Merge to fill missing port coordinates
schedule_data = schedule_data.merge(
    ports_data[['portId', 'latitude', 'longitude']],
    left_on='portId',
    right_on='portId',
    how='left',
    suffixes=('', '_port')
)

# Fill missing portLatitude and portLongitude
schedule_data['portLatitude'].fillna(schedule_data['latitude'], inplace=True)
schedule_data['portLongitude'].fillna(schedule_data['longitude'], inplace=True)

# Drop the temporary columns
schedule_data.drop(columns=['latitude', 'longitude'], inplace=True)

# 5. Dropping columns not needed
schedule_data.drop('portName', axis=1, inplace=True)
schedule_data.drop('shippingLineName', axis=1, inplace=True)



# 7. Save the cleaned dataset
cleaned_file_path = "data/cleaned/clean2/cleaned2_schedules_to_may_2024.csv"  # Replace with your desired path
schedule_data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned schedule dataset saved to: {cleaned_file_path}")
