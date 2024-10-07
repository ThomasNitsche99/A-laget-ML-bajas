import pandas as pd
import numpy as np
from datetime import datetime

# Function to clean the AIS dataset
def clean_ais_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, sep='|')

    # Convert 'time' to datetime format
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Clean COG (Course Over Ground)
    df['cog'] = pd.to_numeric(df['cog'], errors='coerce')
    df['cog'] = np.where((df['cog'] < 0) | (df['cog'] > 360), np.nan, df['cog'])
    
    # Clean SOG (Speed Over Ground)
    df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
    df['sog'] = np.where((df['sog'] < 0) | (df['sog'] > 102.2), np.nan, df['sog'])

    # Clean ROT (Rate of Turn)
    df['rot'] = pd.to_numeric(df['rot'], errors='coerce')
    df['rot'] = np.where((df['rot'] < -128) | (df['rot'] > 127), np.nan, df['rot'])

    # Clean Heading
    df['heading'] = pd.to_numeric(df['heading'], errors='coerce')
    df['heading'] = np.where((df['heading'] < 0) | (df['heading'] > 360), np.nan, df['heading'])

    # Clean NAVSTAT
    df['navstat'] = pd.to_numeric(df['navstat'], errors='coerce')
    df['navstat'] = np.where(df['navstat'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), df['navstat'], np.nan)

    # Clean ETA Raw
    df['etaRaw'] = pd.to_datetime(df['etaRaw'], format='%m-%d %H:%M', errors='coerce')

    # Clean Latitude
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['latitude'] = np.where((df['latitude'] < -90) | (df['latitude'] > 90), np.nan, df['latitude'])

    # Clean Longitude
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['longitude'] = np.where((df['longitude'] < -180) | (df['longitude'] > 180), np.nan, df['longitude'])

    # Drop rows with any NaN values after cleaning
    cleaned_df = df.dropna()

    # Reset index
    cleaned_df.reset_index(drop=True, inplace=True)

    return cleaned_df

# Specify the path to your dataset
file_path = 'data/ais_train.csv'  # Replace with your actual file path

# Call the cleaning function
cleaned_data = clean_ais_data(file_path)

# Save the cleaned data to a new CSV file
cleaned_data.to_csv('data/cleaned_data/cleaned_ais_train1.csv', index=False)

print("Data cleaning complete. Cleaned dataset saved as 'cleaned_ais_data.csv'.")
