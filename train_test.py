import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load the preprocessed data
train_data = pd.read_csv('Data/train_processed.csv')
test_data = pd.read_csv('Data/test_processed.csv')

# Split the data into features and target
X_train = train_data.drop('is_fraud', axis=1)
y_train = train_data['is_fraud']
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud']

# Model performance dictionary
model_performance = {}

# Train and evaluate a Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators = 500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Evaluating Random Forest model...")
print(classification_report(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_rf_probs))
model_performance['Random Forest'] = roc_auc_score(y_test, y_pred_rf_probs)

# Save the model
joblib.dump(rf_model, 'Models/random_forest_model.pkl')

# Train and evaluate an XGBoost model
print("Training XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate = 0.1,
    scale_pos_weight=1.2500265477328236,
    subsample=0.6,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Evaluating XGBoost model...")
print(classification_report(y_test, y_pred_xgb))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_xgb_probs))
model_performance['XGBoost'] = roc_auc_score(y_test, y_pred_xgb_probs)

# Save the model
joblib.dump(xgb_model, 'Models/xgboost_model.pkl')

# Train and evaluate an SVM model
print("Training SVM model...")
svm_fraction = 0.5
X_train_small = X_train.sample(frac=svm_fraction, random_state=42)
y_train_small = y_train.loc[X_train_small.index]

svm_model = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    C = 10,
    gamma='auto',
    random_state=42
)

svm_model.fit(X_train_small, y_train_small)
y_pred_svm = svm_model.predict(X_test)
y_pred_svm_probs = svm_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Evaluating SVM model...")
print(classification_report(y_test, y_pred_svm))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_svm_probs))
model_performance['SVM'] = roc_auc_score(y_test, y_pred_svm_probs)

# Save the model
joblib.dump(svm_model, 'Models/svm_model.pkl')

# Print the model performance
print("\nModel Performance (AUC-ROC):")
for model, auc_roc in model_performance.items():
    print(f"{model}: {auc_roc:.4f}")
