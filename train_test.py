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
    bootstrap = True, 
    ccp_alpha = 0.0, 
    criterion = 'gini', 
    max_depth = 22, 
    max_features = 'sqrt', 
    min_impurity_decrease = 0.0, 
    min_samples_leaf = 1, 
    min_samples_split = 2, 
    min_weight_fraction_leaf = 0.0, 
    n_estimators = 500, 
    oob_score = False, 
    random_state = 42, 
    verbose = 0, 
    warm_start = False
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
    objective = 'binary:logistic', 
    learning_rate = 0.2, 
    max_depth = 12,
    enable_categorical = False,
    eval_metric = 'logloss',
    random_state = 42, 
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
    kernel= 'rbf',
    C = 90,
    cache_size = 200,
    degree = 3,
    decision_function_shape = 'ovr',
    gamma=9e-05,
    max_iter=-1,
    probability=True,
    shrinking=True,
    tol=0.001,
    verbose=False,
    random_state=42
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
    'activation': 'relu', 
    'alpha': 0.0001, 
    'batch_size': 'auto', 
    'beta_1': 0.9, 
    'beta_2': 0.999, 
    'early_stopping': False, 
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (100,), 
    'learning_rate': 'constant', 
    'learning_rate_init': 0.001, 
    'max_fun': 15000, 
    'max_iter': 200, 
    'momentum': 0.9, 
    'n_iter_no_change': 10, 
    'nesterovs_momentum': True, 
    'power_t': 0.5, 
    'random_state': 42, 
    'shuffle': True, 
    'solver': 'adam', 
    'tol': 0.0001,
    'validation_fraction': 0.1, 
    'verbose': False, 
    'warm_start': False,
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
    random_state=42
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