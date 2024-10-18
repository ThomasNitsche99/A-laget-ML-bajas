import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Step 1: Load the AIS dataset
df = pd.read_csv('/Users/eivindmidtbo/Desktop/dev/Maskinlaering_i_praksis/A-laget-ML-bajas/data/ais_train.csv', sep='|')

# Step 2: Convert the 'time' column to datetime format (if needed)
df['time'] = pd.to_datetime(df['time'])

# Step 3: Filter for a specific ship (replace with your desired vesselId)
vessel_id = '61e9f3a8b937134a3c4bfdf7'  # Change this to the vessel you want to track
vessel_data = df[df['vesselId'] == vessel_id]

# Step 4: Set up the map using Cartopy with a specific projection
plt.figure(figsize=(10, 8))
ax_map = plt.axes(projection=ccrs.Mercator())

# Step 5: Ensure the data is sorted by time to draw lines correctly
vessel_data = vessel_data.sort_values('time')

# Step 6: Plot lines between the vessel positions
ax_map.plot(vessel_data['longitude'], vessel_data['latitude'], 
             transform=ccrs.PlateCarree(), 
             linewidth=2, alpha=0.7, color='blue')  # Color for the line

# Step 7: Add small points for each position
ax_map.scatter(vessel_data['longitude'], vessel_data['latitude'], 
               transform=ccrs.PlateCarree(), 
               color='red', s=10, alpha=0.9)  # Small red points

# Add coastlines for context
ax_map.coastlines()

# Optional: Add gridlines for better geographic context
ax_map.gridlines(draw_labels=True)

# Add title
plt.title(f'AIS Vessel Position for Ship: {vessel_id}')

# Show the plot
plt.show()
