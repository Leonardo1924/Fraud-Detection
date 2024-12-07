import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Step 1: Load the test data (kaggle.csv)
test_data = pd.read_csv('Data/kaggle.csv')

# Step 2: Preprocess the test data (based on Data_Preprocessing.py logic)
# Convert 'trans_date_trans_time' to datetime format
test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])

# Transaction Velocity Features
test_data['transactions_last_hour'] = test_data.groupby('cc_num')['trans_date_trans_time'].transform(
    lambda x: x.diff().dt.total_seconds().lt(3600).cumsum().fillna(0)
)

test_data['transactions_last_day'] = test_data.groupby('cc_num')['trans_date_trans_time'].transform(
    lambda x: x.diff().dt.total_seconds().lt(86400).cumsum().fillna(0)
)

# Amount Features
test_data['avg_transaction_amt'] = test_data.groupby('cc_num')['amt'].transform('mean')
test_data['amt_deviation'] = (test_data['amt'] - test_data['avg_transaction_amt']).abs()

# Time-based Features
test_data['transaction_hour'] = test_data['trans_date_trans_time'].dt.hour
test_data['transaction_day'] = test_data['trans_date_trans_time'].dt.dayofweek
test_data['transaction_month'] = test_data['trans_date_trans_time'].dt.month

# Add missing 'age' feature since 'dob' is not available in Kaggle data
test_data['age'] = 0  # Default value since 'dob' does not exist in Kaggle

# Drop unnecessary columns
columns_to_drop = ['trans_date_trans_time', 'merchant', 'trans_num', 'device_os', 'index']
test_data = test_data.drop(columns=columns_to_drop, errors='ignore')

# Step 3: Align features with the training set
# Load the list of features from training
try:
    trained_features = joblib.load('Models/trained_features.pkl')
    print(f"Trained features loaded successfully: {len(trained_features)} features")
except FileNotFoundError:
    raise Exception("Trained features file 'Models/trained_features.pkl' not found. Please ensure it is saved properly during training.")

# Add any missing columns that were present during training but are missing in the Kaggle data
for feature in trained_features:
    if feature not in test_data.columns:
        print(f"Adding missing column: {feature}")
        test_data[feature] = 0  # Default value for missing dummy features

# Drop any extra columns that are not part of the trained features
extra_columns = set(test_data.columns) - set(trained_features)
if extra_columns:
    print(f"Dropping extra columns: {extra_columns}")
    test_data = test_data.drop(columns=extra_columns, errors='ignore')

# Reorder the columns to match the trained model's expected column order
test_data = test_data[trained_features]

# Step 4: Apply scaling to only the features that were scaled during training
# The columns that should be scaled (must match the Data_Preprocessing.py logic)
scale_cols = ['amt', 'age', 'unix_time', 'cc_num', 'transactions_last_hour', 
              'transactions_last_day', 'amt_deviation']

# Scale only the columns that were scaled during training
scaler = StandardScaler()
test_data[scale_cols] = scaler.fit_transform(test_data[scale_cols])

# Step 5: Load the trained models
rf_model = joblib.load('Models/random_forest_model.pkl')
xgb_model = joblib.load('Models/xgboost_model.pkl')
svm_model = joblib.load('Models/svm_model.pkl')
isolation_forest = joblib.load('Models/isolation_forest_model.pkl')

# Step 6: Generate predictions for 'is_fraud' from each model
try:
    rf_probs = rf_model.predict_proba(test_data)[:, 1]
except Exception as e:
    print(f"Error with Random Forest predictions: {e}")
    rf_probs = [0.5] * len(test_data)

try:
    xgb_probs = xgb_model.predict_proba(test_data)[:, 1]
except Exception as e:
    print(f"Error with XGBoost predictions: {e}")
    xgb_probs = [0.5] * len(test_data)

try:
    svm_probs = svm_model.predict_proba(test_data)[:, 1]
except Exception as e:
    print(f"Error with SVM predictions: {e}")
    svm_probs = [0.5] * len(test_data)

try:
    iso_probs = isolation_forest.decision_function(test_data)
    # Normalize Isolation Forest scores to a 0-1 range
    iso_probs = (iso_probs - iso_probs.min()) / (iso_probs.max() - iso_probs.min())
except Exception as e:
    print(f"Error with Isolation Forest predictions: {e}")
    iso_probs = [0.5] * len(test_data)

# Step 7: Combine predictions (simple average of the 4 models)
combined_probs = (rf_probs + xgb_probs + svm_probs + iso_probs) / 4

combined_probs = [round(prob, 2) for prob in combined_probs]  # Round to 3 decimal places

# Step 8: Create a submission file
submission_df = pd.DataFrame({
    'index': range(30000, 30000 + len(combined_probs)),  # Use the original index from kaggle.csv
    'is_fraud': combined_probs
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' generated successfully.")
