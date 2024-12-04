import pandas as pd
import joblib

# Load the dataset
kaggle_dataset_path = 'Data/kaggle.csv'
kaggle_test_data = pd.read_csv(kaggle_dataset_path)

kaggle_test_data['transaction_hour'] = pd.to_datetime(kaggle_test_data['trans_date_trans_time']).dt.hour
kaggle_test_data['transaction_day'] = pd.to_datetime(kaggle_test_data['trans_date_trans_time']).dt.dayofweek
kaggle_test_data['transaction_month'] = pd.to_datetime(kaggle_test_data['trans_date_trans_time']).dt.month

kaggle_test_data = pd.get_dummies(kaggle_test_data, columns=['merchant'], drop_first=True)

trained_model_features = joblib.load('Models/trained_features.pkl')  # Load the features used during training
for feature in trained_model_features:
    if feature not in kaggle_test_data.columns:
        kaggle_test_data[feature] = 0  # Add missing features with zeros

X_kaggle_test = kaggle_test_data[trained_model_features]

best_model = joblib.load('Models/random_forest_model.pkl')  # Load the best model

kaggle_predictions = best_model.predict_proba(X_kaggle_test)[:, 1]

submission = pd.DataFrame({
    'index': kaggle_test_data['index'],  # Ensure 'index' column is preserved
    'is_fraud': kaggle_predictions
})

submission.to_csv('kaggle_submission.csv', index=False)
print("Submission file saved as 'kaggle_submission.csv'.")