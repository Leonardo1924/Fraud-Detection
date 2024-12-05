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

def clean_data(input_path, output_path, tittle_suffix):
    """
    Cleans the dataset and saves the cleaned data with visualizations.

    Parameters:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the cleaned dataset.
        title_suffix (str): Suffix for plot titles (e.g., 'Train' or 'Test').

    Returns:
        None
    """
    data = pd.read_csv(input_path)

    # Plot Missing Values
    missing_values = (data.isnull().sum() / len(data)) * 100
    plt.figure(figsize=(12, 6))
    plt.bar(missing_values.index, missing_values.values, color=colors)
    plt.title(f"Missing Values Percentage ({tittle_suffix} Data)", fontsize=16, fontweight="bold")
    plt.ylabel("Percentage", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    plt.show()

    columns_to_remove = columns_to_remove = ['first', 'last', 'merchant', 'index', 'trans_num', 'merchant_id', 'lat', 'long', 'device_os', 'city_pop', 'state']
    data = data.drop(columns=columns_to_remove, errors='ignore')

    # Handle Missing Values
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'], errors='coerce')
    median_date = data['trans_date_trans_time'].median()
    data['trans_date_trans_time'] = data['trans_date_trans_time'].fillna(median_date)

    data['amt'] = data['amt'].fillna(data['amt'].median())
    data['category'] = data['category'].fillna(data['category'].mode()[0])

    data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
    data['dob'] = data['dob'].fillna(data['dob'].median())

    categorical_cols = ['gender', 'street', 'city', 'job']
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    geo_cols = ['merch_lat', 'merch_long', 'zip']
    for col in geo_cols:
        data[col] = data[col].fillna(data[col].median())

    # Find restant missing values
    missing_values = (data.isnull().sum() / len(data)) * 100
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print(f"Remaining missing values:\n{missing_values}")
    else:
        print("All missing values handled.")

    data.to_csv(output_path, index=False)
    print(f"Data cleaned and saved to {output_path}")

if __name__ == "__main__":
    clean_data("Data/train_data.csv", "Data/train_cleaned.csv", "Train Dataset")
    clean_data("Data/test_data.csv", "Data/test_cleaned.csv", "Test Dataset")