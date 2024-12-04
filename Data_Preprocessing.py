import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier



dataset_path = 'Data/cleaned_dataset.csv'
dataset = pd.read_csv(dataset_path)

# Univariate Analysis - Distribution of Transaction Amounts
plt.figure(figsize=(10, 6))
sns.histplot(dataset['amt'], bins=30, kde=True, color='blue')
plt.title('Distribution of Transaction Amounts', fontsize=16)
plt.xlabel('Transaction Amount (€)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Bivariate Analysis - Fraud vs. Transaction Amount
plt.figure(figsize=(10, 6))
sns.boxplot(data=dataset, x='is_fraud', y='amt', hue='is_fraud', palette='Set2', dodge=False)
plt.title('Transaction Amounts by Fraud Status', fontsize=16)
plt.xlabel('Fraud Status (0: Legitimate, 1: Fraudulent)', fontsize=12)
plt.ylabel('Transaction Amount (€)', fontsize=12)
plt.legend([], [], frameon=False)  # Disable legend
plt.show()

# Count plot of Fraud Status
plt.figure(figsize=(8, 6))
sns.countplot(data=dataset, x='is_fraud', hue='is_fraud', palette='Set1', dodge=False)
plt.title('Fraud Status Count', fontsize=16)
plt.xlabel('Fraud Status (0: Legitimate, 1: Fraudulent)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend([], [], frameon=False)  # Disable legend
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(12, 8))
numeric_cols = dataset.select_dtypes(include=['float64', 'int64']).columns  # Select numeric columns
correlation_matrix = dataset[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

dataset['transaction_hour'] = pd.to_datetime(dataset['trans_date_trans_time']).dt.hour
dataset['transaction_day'] = pd.to_datetime(dataset['trans_date_trans_time']).dt.dayofweek
dataset['transaction_month'] = pd.to_datetime(dataset['trans_date_trans_time']).dt.month

current_year = pd.to_datetime('today').year
dataset['age'] = current_year - pd.to_datetime(dataset['dob']).dt.year

customer_avg = dataset.groupby('cc_num')['amt'].transform('mean')
dataset['amt_vs_customer_avg'] = dataset['amt'] / customer_avg

dataset = pd.get_dummies(dataset, columns=['category', 'gender', 'job','merchant'], drop_first=True)

scale_cols = ['amt', 'age', 'amt_vs_customer_avg']
scaler = StandardScaler()
dataset[scale_cols] = scaler.fit_transform(dataset[scale_cols])

X = dataset.drop(['is_fraud', 'trans_date_trans_time', 'dob', 'trans_num', 'first', 'last', 'street', 'city'], axis=1)
y = dataset['is_fraud']


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

final_dataset = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='is_fraud')], axis=1)

final_dataset.to_csv('Data/processed_dataset.csv', index=False)
print("Dataset scaled, balanced, and saved as 'processed_dataset.csv'.")
