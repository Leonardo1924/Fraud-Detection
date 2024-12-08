import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
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
cv_strategy = StratifiedKFold(n_splits=4, shuffle=True, random_state=666)

# -------------------------
# Random Forest
# -------------------------
try:
    print("\nTuning Random Forest (Randomized Search)...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'max_features': ['sqrt', 'log2', None],
        'max_leaf_nodes': [10, 20, 30, None],
    }
    rf_random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=666),
        param_distributions=rf_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1,
        n_iter=20,
        random_state=666
    )
    rf_random_search.fit(X_train, y_train)

    print("\nBest Random Forest Parameters from Randomized Search:", rf_random_search.best_params_)

    rf_grid_param_grid = {
        'n_estimators': [rf_random_search.best_params_['n_estimators'] - 50, rf_random_search.best_params_['n_estimators'], rf_random_search.best_params_['n_estimators'] + 50],
        'max_depth': [rf_random_search.best_params_['max_depth'] - 2, rf_random_search.best_params_['max_depth'], rf_random_search.best_params_['max_depth'] + 2]
    }
    rf_grid_search = GridSearchCV(
        RandomForestClassifier(random_state=666),
        param_grid=rf_grid_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1
    )
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    print("\nBest Random Forest Parameters from Grid Search:", rf_grid_search.best_params_)
except Exception as e:
    print(f"Error during Random Forest tuning: {e}")

# -------------------------
# XGBoost
# -------------------------
try:
    print("\nTuning XGBoost (Randomized Search)...")
    xgb_param_grid = {
        'max_depth': [3, 5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5, 7],
        'colsample_bytree': [0.3, 0.5, 0.7, 0.9]
    }
    xgb_random_search = RandomizedSearchCV(
        XGBClassifier(random_state=666, eval_metric='logloss'),
        param_distributions=xgb_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1,
        n_iter=20,
        random_state=666
    )
    xgb_random_search.fit(X_train, y_train)

    print("\nBest XGBoost Parameters from Randomized Search:", xgb_random_search.best_params_)

    xgb_grid_param_grid = {
        'learning_rate': [xgb_random_search.best_params_['learning_rate'] - 0.02, xgb_random_search.best_params_['learning_rate'], xgb_random_search.best_params_['learning_rate'] + 0.02],
        'max_depth': [xgb_random_search.best_params_['max_depth'] - 2, xgb_random_search.best_params_['max_depth'], xgb_random_search.best_params_['max_depth'] + 2]
    }
    xgb_grid_search = GridSearchCV(
        XGBClassifier(random_state=666, eval_metric='logloss'),
        param_grid=xgb_grid_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1
    )
    xgb_grid_search.fit(X_train, y_train)
    best_xgb_model = xgb_grid_search.best_estimator_
    print("\nBest XGBoost Parameters from Grid Search:", xgb_grid_search.best_params_)
except Exception as e:
    print(f"Error during XGBoost tuning: {e}")

# -------------------------
# SVM
# -------------------------
try:
    print("\nTuning SVM (Randomized Search)...")
    svm_fraction = 0.5
    X_train_small = X_train.sample(frac=svm_fraction, random_state=666)
    y_train_small = y_train.loc[X_train_small.index]
    svm_param_grid = {
        'C': np.logspace(-2, 2, 5),
        'gamma': np.logspace(-4, -1, 4)
    }
    svm_random_search = RandomizedSearchCV(
        SVC(probability=True, random_state=666),
        param_distributions=svm_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1,
        n_iter=16,
        random_state=666
    )
    svm_random_search.fit(X_train_small, y_train_small)

    print("\nBest SVM Parameters from Randomized Search:", svm_random_search.best_params_)

    svm_grid_param_grid = {
        'C': [svm_random_search.best_params_['C'] * 0.9, svm_random_search.best_params_['C'], svm_random_search.best_params_['C'] * 1.1],
        'gamma': [svm_random_search.best_params_['gamma'] * 0.9, svm_random_search.best_params_['gamma'], svm_random_search.best_params_['gamma'] * 1.1]
    }
    svm_grid_search = GridSearchCV(
        SVC(probability=True, random_state=666),
        param_grid=svm_grid_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1
    )
    svm_grid_search.fit(X_train_small, y_train_small)
    best_svm_model = svm_grid_search.best_estimator_
    print("\nBest SVM Parameters from Grid Search:", svm_grid_search.best_params_)
except Exception as e:
    print(f"Error during SVM tuning: {e}")

# -------------------------
# Neural Network
# -------------------------
try:
    print("\nTuning MLP (Randomized Search)...")
    mlp_param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 100)],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01]
    }
    mlp_random_search = RandomizedSearchCV(
        MLPClassifier(random_state=666),
        param_distributions=mlp_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1,
        n_iter=20,
        random_state=666
    )
    mlp_random_search.fit(X_train, y_train)

    print("\nBest MLP Parameters from Randomized Search:", mlp_random_search.best_params_)

    mlp_grid_param_grid = {
        'alpha': [mlp_random_search.best_params_['alpha'] * 0.9, mlp_random_search.best_params_['alpha'], mlp_random_search.best_params_['alpha'] * 1.1]
    }
    mlp_grid_search = GridSearchCV(
        MLPClassifier(random_state=666),
        param_grid=mlp_grid_param_grid,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=3,
        n_jobs=-1
    )
    mlp_grid_search.fit(X_train, y_train)
    best_mlp_model = mlp_grid_search.best_estimator_
    print("\nBest MLP Parameters from Grid Search:", mlp_grid_search.best_params_)
except Exception as e:
    print(f"Error during MLP tuning: {e}")

# -------------------------
#Save the best parameters
# -------------------------
best_params = []
try:
    best_params.append(best_rf_model.get_params())
    best_params.append(best_xgb_model.get_params())
    best_params.append(best_svm_model.get_params())
    best_params.append(best_mlp_model.get_params())
except Exception as e:
    print(f"Error during saving best parameters: {e}")

print("\nBest Parameters for each model:")
for i, model in enumerate(['Random Forest', 'XGBoost', 'SVM', 'MLP']):
    print(f"{model}: {best_params[i]}")

