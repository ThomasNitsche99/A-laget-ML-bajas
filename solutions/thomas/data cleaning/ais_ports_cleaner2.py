import pandas as pd

# Load the Ports dataset
ports_file_path = "data/formatted/ais_ports_converted.csv"  # Replace with the path to the Ports dataset
ports_data = pd.read_csv(ports_file_path)

# Remove invalid Latitude and Longitude values for ports
ports_data = ports_data[
    ((ports_data['latitude'] >= -90) & (ports_data['latitude'] <= 90)) | (ports_data['latitude'].isnull())
]

ports_data = ports_data[
    ((ports_data['longitude'] >= -180) & (ports_data['longitude'] <= 180)) | (ports_data['longitude'].isnull())
]

#Dropping columns not needed
ports_data.drop('UN_LOCODE', axis=1, inplace=True)
ports_data.drop('ISO', axis=1, inplace=True)
ports_data.drop('countryName', axis=1, inplace=True)
ports_data.drop('portLocation', axis=1, inplace=True)
# ports_data.drop('name', axis=1, inplace=True)


# Save the cleaned Ports dataset
cleaned_ports_file_path = "data/cleaned/clean2/cleaned2_ports_dataset.csv"
ports_data.to_csv(cleaned_ports_file_path, index=False)

print(f"Cleaned Ports dataset saved to: {cleaned_ports_file_path}")
