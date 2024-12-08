import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

# ---------------------------------
# Random Forest
# ---------------------------------

# Train and evaluate a Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators = 550,
    max_depth=22,
    class_weight='balanced',
    random_state=666
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

# ---------------------------------
# XGBoost
# ---------------------------------

# Train and evaluate an XGBoost model
print("Training XGBoost model...")
xgb_model = XGBClassifier(
    max_depth=12,
    learning_rate=0.22,
    min_child_weight=1,
    colsample_bytree=0.3,
    random_state=666
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

# ---------------------------------
# SVM
# ---------------------------------

# Train and evaluate an SVM model
print("Training SVM model...")
svm_model = SVC(
    kernel='rbf',
    probability=True,
    C = 90,
    gamma=0.0001,
    class_weight='balanced',
    random_state=666
)

svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_pred_svm_probs = svm_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Evaluating SVM model...")
print(classification_report(y_test, y_pred_svm))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_svm_probs))
model_performance['SVM'] = roc_auc_score(y_test, y_pred_svm_probs)

# Save the model
joblib.dump(svm_model, 'Models/svm_model.pkl')

# ---------------------------------
# Neural Network (MLP Classifier)
# ---------------------------------

print("Training Neural Network model...")
best_mlp_params = {
    'hidden_layer_sizes': (100,), 
    'activation': 'tanh',
    'solver': 'adam',
    'alpha': 0.001,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.01,
    'batch_size': 32,
    'max_iter': 500,
    'early_stopping': False,
    'momentum': 0.9,
}

# Create and train MLPClassifier with best parameters
mlp_model = MLPClassifier(
    hidden_layer_sizes=best_mlp_params['hidden_layer_sizes'],
    activation=best_mlp_params['activation'],
    solver=best_mlp_params['solver'],
    alpha=best_mlp_params['alpha'],
    learning_rate=best_mlp_params['learning_rate'],
    learning_rate_init=best_mlp_params['learning_rate_init'],
    batch_size=best_mlp_params['batch_size'],
    max_iter=best_mlp_params['max_iter'],
    early_stopping=best_mlp_params['early_stopping'],
    momentum=best_mlp_params['momentum'],
    random_state=666
)

# Fit the model
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
y_pred_mlp_probs = mlp_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nEvaluating Neural Network (MLP) model...")
print(classification_report(y_test, y_pred_mlp, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_mlp_probs))
model_performance['MLP'] = roc_auc_score(y_test, y_pred_mlp_probs)

# Save the model
joblib.dump(mlp_model, 'Models/mlp_model.pkl')

# Print the model performance
print("\nModel Performance (AUC-ROC):")
for model, auc_roc in model_performance.items():
    print(f"{model}: {auc_roc:.4f}")

# Save the trained features
joblib.dump(X_train.columns, 'Models/trained_features.pkl')

# Check feature importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})
print("\nFeature Importance (Random Forest):")
print(feature_importance.sort_values('Importance', ascending=False).head(10))

# Check feature importance for XGBoost
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
})
print("\nFeature Importance (XGBoost):")
print(feature_importance.sort_values('Importance', ascending=False).head(10))

# Check feature importance for Neural Network
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': mlp_model.coefs_[0].mean(axis=1)
})
print("\nFeature Importance (Neural Network):")
print(feature_importance.sort_values('Importance', ascending=False).head(10))