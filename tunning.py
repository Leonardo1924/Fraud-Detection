import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import numpy as np

# Load the preprocessed data
train_data = pd.read_csv('Data/train_processed.csv')
test_data = pd.read_csv('Data/test_processed.csv')

# Split the data into features and target
X_train = train_data.drop('is_fraud', axis=1)
y_train = train_data['is_fraud']
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud']

# Define cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)

# Random Forest Tuning
try:
    print("\nTuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_depth': [3, 6, 9],
        'max_features': [None, 'sqrt', 'log2'],  # Fixed 'None' to None
        'max_leaf_nodes': [3, 6, 9],
    }
    rf_grid_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=666),
        param_distributions=rf_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1,
        n_iter=10,  # Increase for better exploration
        random_state=666
    )
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    print("\nBest Random Forest Parameters:", rf_grid_search.best_params_)
except Exception as e:
    print(f"Error during Random Forest tuning: {e}")

# XGBoost Tuning
try:
    print("\nTuning XGBoost...")
    xgb_param_grid = {
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'min_child_weight': [1, 3, 5, 7],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4]
    }
    xgb_grid_search = RandomizedSearchCV(
        XGBClassifier(random_state=666, eval_metric='logloss'),
        param_distributions=xgb_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1,
        n_iter=10,  # Increase for better exploration
        random_state=666
    )
    xgb_grid_search.fit(X_train, y_train)
    best_xgb_model = xgb_grid_search.best_estimator_
    print("\nBest XGBoost Parameters:", xgb_grid_search.best_params_)
except Exception as e:
    print(f"Error during XGBoost tuning: {e}")

# SVM Tuning
try:
    print("\nTuning SVM...")
    svm_param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf'],
    }
    X_train_small = X_train.sample(frac=0.2, random_state=666)
    y_train_small = y_train.loc[X_train_small.index]
    svm_grid_search = RandomizedSearchCV(
        SVC(probability=True, random_state=666),
        param_distributions=svm_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,  # Fixed to use cross-validation strategy
        verbose=3,
        n_jobs=-1,
        n_iter=10,  # Increase for better exploration
        random_state=666
    )
    svm_grid_search.fit(X_train_small, y_train_small)
    best_svm_model = svm_grid_search.best_estimator_
    print("\nBest SVM Parameters:", svm_grid_search.best_params_)
except Exception as e:
    print(f"Error during SVM tuning: {e}")

try:
    print("\nNeural Network Tuning...")
    mlp_param_grid = {
    'hidden_layer_sizes': [(100,), (50, 50), (100, 100), (128, 64), (256, 128)],  # Size of hidden layers
    'activation': ['relu', 'tanh', 'logistic'],  # Activation functions
    'solver': ['adam', 'sgd', 'lbfgs'],  # Optimization solvers
    'alpha': [0.0001, 0.001, 0.01, 0.1],  # L2 regularization term
    'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate schedule
    'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
    'batch_size': [32, 64, 128, 256],  # Batch size for stochastic optimizers
    'max_iter': [200, 500, 1000],  # Maximum number of iterations
    'early_stopping': [True, False],  # Early stopping option
    'momentum': [0.9, 0.95, 0.99],  # Momentum for the 'sgd' optimizer
}
    mlp_grid_search = RandomizedSearchCV(
        MLPClassifier(random_state=666),
        param_distributions=mlp_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1,
        n_iter=10,
        random_state=666
    )
    mlp_grid_search.fit(X_train, y_train)
    best_mlp_model = mlp_grid_search.best_estimator_
    print("\nBest Neural Network Parameters:", mlp_grid_search.best_params_)
except Exception as e:
    print(f"Error during Neural Network tuning: {e}")

# Save the parameters
try:
    print("\nSaving the best parameters from each model...")
    parameters = {}
    if 'rf_grid_search' in locals() and hasattr(rf_grid_search, 'best_params_'):
        parameters['Random Forest'] = rf_grid_search.best_params_
    else:
        parameters['Random Forest'] = None
        
    if 'xgb_grid_search' in locals() and hasattr(xgb_grid_search, 'best_params_'):
        parameters['XGBoost'] = xgb_grid_search.best_params_
    else:
        parameters['XGBoost'] = None
        
    if 'svm_grid_search' in locals() and hasattr(svm_grid_search, 'best_params_'):
        parameters['SVM'] = svm_grid_search.best_params_
    else:
        parameters['SVM'] = None

    if 'mlp_grid_search' in locals() and hasattr(mlp_grid_search, 'best_params_'):
        parameters['Neural Network'] = mlp_grid_search.best_params_
    else:
        parameters['Neural Network'] = None

    for model_name, params in parameters.items():
        print(f"\n{model_name} Best Parameters:")
        print(params)
except Exception as e:
    print(f"Error while saving best parameters: {e}")