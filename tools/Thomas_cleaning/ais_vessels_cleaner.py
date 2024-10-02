import pandas as pd

# Load the vessels dataset
vessels_file_path = "data/formatted/vessels_converted.csv"
vessels_data = pd.read_csv(vessels_file_path)

#Temporary index for storing orignal ordering of lines
vessels_data['original_index'] = vessels_data.index

# 1. Validate and handle numeric columns
numeric_columns = ['CEU', 'DWT', 'GT', 'NT', 'breadth', 'depth', 'draft', 'enginePower', 'freshWater', 'fuel', 'length', 'maxHeight', 'maxSpeed', 'maxWidth', 'rampCapacity']

# Remove rows with negative or unrealistic values in numeric columns
for col in numeric_columns:
    vessels_data = vessels_data[(vessels_data[col].isnull()) | (vessels_data[col] >= 0)]

# 2. Handle missing values within the context of `vesselId`
# Group the data by `vesselId` and fill missing values within each group
def fill_missing_values_within_group(group):
    # Replace missing numeric values with the median for each vessel group
    for col in numeric_columns:
        group[col] = group[col].fillna(group[col].median())
    return group

# Apply the function to each vesselId group
vessels_data = vessels_data.groupby('vesselId').apply(fill_missing_values_within_group)

# Drop the group index to restore the original structure
vessels_data.reset_index(drop=True, inplace=True)


# Fill missing values in categorical columns with "Unknown"
categorical_columns = ['shippingLineId', 'vesselId', 'vesselType', 'homePort']
for col in categorical_columns:
    vessels_data[col].fillna('Unknown', inplace=True)

# 3. Remove duplicates
vessels_data.drop_duplicates(inplace=True)


# 5. Restore the original order based on the 'original_index' column
vessels_data.sort_values(by='original_index', inplace=True)
vessels_data.drop(columns=['original_index'], inplace=True)  # Remove the temporary index column


# 6. Save the cleaned vessels dataset
cleaned_vessels_file_path = "data/formatted/cleaned_vessels.csv"
vessels_data.to_csv(cleaned_vessels_file_path, index=False)

print(f"Cleaned vessels dataset saved to: {cleaned_vessels_file_path}")
