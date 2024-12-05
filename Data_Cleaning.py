import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

colors = [
    "#ffb347", "#ffd1dc", "#b19cd9", "#ffcccb", "#fdfd96", 
    "#cfcfc4", "#ffb6c1", "#c3b091", "#d8bfd8", "#add8e6",
    "#90ee90", "#f08080", "#fafad2", "#dda0dd", "#ffc0cb",
    "#ffdab9", "#e6e6fa", "#98fb98", "#ffe4e1", "#afeeee",
    "#f5deb3", "#ffdead", "#e0ffff", "#d3ffce", "#ff69b4"
]

# Load the dataset
data_path = 'Data/data.csv'
dataset = pd.read_csv(data_path)

# Initial information about the dataset
dataset.head(), dataset.info()

# Calculate missing value percentages
missing_values = (dataset.isnull().sum() / len(dataset)) * 100

# Plot missing values
plt.figure(figsize=(12, 6))
bars = plt.bar(missing_values.index, missing_values.values, color=colors)
plt.title('Missing Values by Column', fontsize=16, fontweight='bold')
plt.ylabel('Count of Missing Values', fontsize=12)
plt.xlabel('Feature', fontsize=12)
plt.xticks(rotation=90, fontsize=10)
plt.tight_layout()
plt.show()

# Remove columns with more than the threshold of missing values
threshold = 50
columns_to_keep = missing_values[missing_values <= threshold].index
dataset = dataset[columns_to_keep]

# Handle `trans_date_trans_time`
dataset['trans_date_trans_time'] = pd.to_datetime(dataset['trans_date_trans_time'], errors='coerce')
median_date = dataset['trans_date_trans_time'].median()
dataset['trans_date_trans_time'] = dataset['trans_date_trans_time'].fillna(median_date)
dataset['trans_date_trans_time'] = dataset['trans_date_trans_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Handle numerical column `amt`
dataset['amt'] = dataset['amt'].fillna(dataset['amt'].median())

# Handle categorical column `category`
dataset['category'] = dataset['category'].fillna(dataset['category'].mode()[0])

# Handle `merchant_id` with a placeholder (e.g., -1)
dataset['merchant_id'] = dataset['merchant_id'].fillna(-1)

# Handle `dob`
dataset['dob'] = pd.to_datetime(dataset['dob'], errors='coerce')
median_dob = dataset['dob'].median()
dataset['dob'] = dataset['dob'].fillna(median_dob)

# Handle categorical columns with mode
categorical_cols = ['first', 'last', 'gender', 'street', 'city', 'job']
for col in categorical_cols:
    dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

# Handle geographical columns with median
geo_cols = ['merch_lat', 'merch_long', 'zip']
for col in geo_cols:
    dataset[col] = dataset[col].fillna(dataset[col].median())

# Remove unnecessary columns
columns_to_remove = ['first', 'last', 'merchant_id', 'merchant', 'index', 'trans_num']
dataset = dataset.drop(columns=columns_to_remove, errors='ignore')  # errors='ignore' ensures no error if column not found

# Final check for missing values
print("Remaining Missing Values After Imputation:\n", dataset.isnull().sum())

dataset.to_csv('Data/cleaned_dataset.csv', index=False)
print("Cleaned dataset saved as 'cleaned_dataset.csv'")
