import pandas as pd

# Load the uploaded dataset
file_path = 'sentimentdataset.csv'
data = pd.read_csv(file_path)

# Remove unnecessary columns
data_cleaned = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

# Check for any missing values
missing_values = data_cleaned.isnull().sum()

# Display the cleaned data structure and missing values
print(data_cleaned.head())
print(missing_values)
