import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np

def plot_reduced_correlation_matrix(data, relevant_columns, title="Reduced Correlation Matrix"):
    """
    Plots a reduced correlation matrix heatmap for selected columns.
    
    Parameters:
        data (pd.DataFrame): The input dataframe.
        relevant_columns (list): List of columns to include in the correlation matrix.
        title (str): Title of the plot.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    corr_matrix = data[relevant_columns].corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot the heatmap with the mask
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(title, fontsize=14)
    plt.show()

def preprocess_train_data(input_file, output_file):
    """
    Preprocess the training dataset with SMOTE and save the processed data.

    Parameters:
        input_file (str): Path to the cleaned training dataset.
        output_file (str): Path to save the processed training dataset.

    Returns:
        None
    """
    # Load the data
    data = pd.read_csv(input_file)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(data['amt'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Transaction Amount', fontsize = 12)
    plt.xlabel('Transaction Amount (€)', fontsize=12)
    plt.ylabel('Frequency', fontsize = 12)
    #plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='is_fraud', hue='is_fraud', palette='Set1', dodge=False, legend=False)
    plt.title('Fraud Status Count', fontsize=16)
    plt.xlabel('Fraud Status', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    #plt.show()

    # Feature engineering
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

    # Transaction Velocity Feature
    data['transactions_last_hour'] = data.groupby('cc_num')['trans_date_trans_time'].transform(
        lambda x: x.diff().dt.total_seconds().lt(3600).cumsum().fillna(0))
    
    data['transactions_last_day'] = data.groupby('cc_num')['trans_date_trans_time'].transform(
        lambda x: x.diff().dt.total_seconds().lt(86400).cumsum().fillna(0))
    
    data['avg_transaction_amt'] = data.groupby('cc_num')['amt'].transform('mean')
    data['amt_deviation'] = np.abs(data['amt'] - data['avg_transaction_amt'])
    
    data['transaction_hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
    data['transaction_day'] = pd.to_datetime(data['trans_date_trans_time']).dt.dayofweek
    data['transaction_month'] = pd.to_datetime(data['trans_date_trans_time']).dt.month

    current_year = pd.to_datetime('today').year
    data['age'] = current_year - pd.to_datetime(data['dob']).dt.year

    data = data.drop(['trans_date_trans_time', 'dob', 'city'], axis=1)
    
    data = pd.get_dummies(data, columns=['category', 'gender', 'job', 'street'], drop_first=True)

    numeric_columns = ['amt', 'age', 'transactions_last_hour', 'transactions_last_day',
                   'amt_deviation', 'is_fraud']
    
    print("Correlation Matrix Before Preprocessing:")
    plot_reduced_correlation_matrix(data, numeric_columns, title="Reduced Correlation Matrix")

    # Standardize the data
    scale_cols = ['amt', 'age', 'unix_time', 'cc_num', 'transactions_last_hour', 'transactions_last_day',
                  'amt_deviation']
    scaler = StandardScaler()
    data[scale_cols] = scaler.fit_transform(data[scale_cols])

    # Apply SMOTE
    X = data.drop(['is_fraud'], axis=1)
    y = data['is_fraud']

    smote = SMOTE(sampling_strategy=1.0, random_state=666)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    processed_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='is_fraud')], axis=1)

    processed_data.to_csv(output_file, index=False)
    print(f"Training data processed and saved to {output_file}")

    # Load the original and augmented datasets
    original_train_data = pd.read_csv('Data/train_cleaned.csv')
    augmented_train_data = pd.read_csv('Data/train_processed.csv')
    
    # Plotting class distribution before SMOTE
    original_counts = Counter(original_train_data['is_fraud'])
    plt.bar(original_counts.keys(), original_counts.values(), color='blue', alpha=0.6, label="Before SMOTE")
    
    # Plotting class distribution after SMOTE
    augmented_counts = Counter(augmented_train_data['is_fraud'])
    plt.bar(augmented_counts.keys(), augmented_counts.values(), color='red', alpha=0.6, label="After SMOTE")
    
    plt.title("Class Distribution Before and After SMOTE")
    plt.xlabel("Class (is_fraud)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    # Display the increase in minority class
    minority_class_increase = augmented_counts[1] - original_counts[1]
    print(f"\nMinority class increased by: {minority_class_increase}")

def preprocess_test_data(input_file, output_file):
    """
    Preprocess the testing dataset without SMOTE and save the processed data.

    Parameters:
        input_file (str): Path to the cleaned testing dataset.
        output_file (str): Path to save the processed testing dataset.

    Returns:
        None
    """
    # Load the data
    data = pd.read_csv(input_file)
    
    # Feature engineering
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

    # Transaction Velocity Feature
    data['transactions_last_hour'] = data.groupby('cc_num')['trans_date_trans_time'].transform(
        lambda x: x.diff().dt.total_seconds().lt(3600).cumsum().fillna(0))
    
    data['transactions_last_day'] = data.groupby('cc_num')['trans_date_trans_time'].transform(
        lambda x: x.diff().dt.total_seconds().lt(86400).cumsum().fillna(0))
    
    data['avg_transaction_amt'] = data.groupby('cc_num')['amt'].transform('mean')
    data['amt_deviation'] = np.abs(data['amt'] - data['avg_transaction_amt'])
    
    data['transaction_hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
    data['transaction_day'] = pd.to_datetime(data['trans_date_trans_time']).dt.dayofweek
    data['transaction_month'] = pd.to_datetime(data['trans_date_trans_time']).dt.month

    current_year = pd.to_datetime('today').year
    data['age'] = current_year - pd.to_datetime(data['dob']).dt.year

    data = data.drop(['trans_date_trans_time', 'dob', 'city'], axis=1)
    
    data = pd.get_dummies(data, columns=['gender'], drop_first=True)
    data = pd.get_dummies(data, columns=['category', 'job', 'street'], drop_first=False)

    # Standardize the data
    scale_cols = ['amt', 'age', 'unix_time', 'cc_num', 'transactions_last_hour', 'transactions_last_day',
                  'amt_deviation']
    scaler = StandardScaler()
    data[scale_cols] = scaler.fit_transform(data[scale_cols])

    data.to_csv(output_file, index=False)
    print(f"Testing data processed and saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    preprocess_train_data('Data/train_cleaned.csv', 'Data/train_processed.csv')
    preprocess_test_data('Data/test_cleaned.csv', 'Data/test_processed.csv')


