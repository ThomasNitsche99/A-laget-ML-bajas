import pandas as pd

# Load the Ports dataset
ports_file_path = "data/formatted/ais_ports_converted.csv"  # Replace with the path to the Ports dataset
ports_data = pd.read_csv(ports_file_path)

# Remove invalid Latitude and Longitude values for ports
ports_data = ports_data[(ports_data['latitude'] >= -90) & (ports_data['latitude'] <= 90) | (ports_data['latitude'].isnull())]
ports_data = ports_data[(ports_data['longitude'] >= -180) & (ports_data['longitude'] <= 180) | (ports_data['longitude'].isnull())]

# Handle missing values in Ports dataset
ports_data['UN_LOCODE'].fillna('Unknown', inplace=True)
ports_data['name'].fillna('Unknown', inplace=True)
ports_data['portLocation'].fillna('Unknown', inplace=True)
ports_data['countryName'].fillna('Unknown', inplace=True)
ports_data['ISO'].fillna('Unknown', inplace=True)

# Remove duplicate rows in the Ports dataset
ports_data = ports_data.drop_duplicates()

# Save the cleaned Ports dataset
cleaned_ports_file_path = "data/cleaned/cleaned_ports_dataset.csv"
ports_data.to_csv(cleaned_ports_file_path, index=False)

print(f"Cleaned Ports dataset saved to: {cleaned_ports_file_path}")
