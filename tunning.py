import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import IsolationForest

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
print("Tuning Random Forest...")
rf_param_grid = {
    'n_estimators': [400, 600, 700],
    'max_depth': [15, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample']
}
rf_grid_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=666),
    param_distributions=rf_param_grid,
    scoring='roc_auc',
    cv=cv_strategy,
    verbose=2,
    n_jobs=-1,
    n_iter=20,  # Number of parameter combinations to try
    random_state=666
)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
print("Best Random Forest Parameters:", rf_grid_search.best_params_)

# XGBoost Tuning
print("Tuning XGBoost...")
xgb_param_grid = {
    'n_estimators': [400, 600, 700],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
}
xgb_grid_search = RandomizedSearchCV(
    XGBClassifier(random_state=666),
    param_distributions=xgb_param_grid,
    scoring='roc_auc',
    cv=cv_strategy,
    verbose=2,
    n_jobs=-1,
    n_iter=20,
    random_state=666
)
xgb_grid_search.fit(X_train, y_train)
best_xgb_model = xgb_grid_search.best_estimator_
print("Best XGBoost Parameters:", xgb_grid_search.best_params_)

# SVM Tuning
print("Tuning SVM...")
svm_param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly'],
    'class_weight': ['balanced']
}
X_train_small = X_train.sample(frac=0.5, random_state=666)
y_train_small = y_train.loc[X_train_small.index]
svm_grid_search = RandomizedSearchCV(
    SVC(probability=True, random_state=666),
    param_distributions=svm_param_grid,
    scoring='roc_auc',
    cv=cv_strategy,
    verbose=2,
    n_jobs=-1,
    n_iter=8,
    random_state=666
)
svm_grid_search.fit(X_train_small, y_train_small)
best_svm_model = svm_grid_search.best_estimator_
print("Best SVM Parameters:", svm_grid_search.best_params_)

# Isolation Forest Tuning
print("Tuning Isolation Forest...")
iso_param_grid = {
    'n_estimators': [400, 500, 600],
    'max_samples': [0.1, 0.5, 1.0],
    'contamination': [0.05, 0.1, 0.2]
}
iso_grid_search = RandomizedSearchCV(
    IsolationForest(random_state=666),
    param_distributions=iso_param_grid,
    scoring='roc_auc',
    cv=cv_strategy,
    verbose=2,
    n_jobs=-1,
    n_iter=20,
    random_state=666
)
iso_grid_search.fit(X_train, y_train)
best_iso_model = iso_grid_search.best_estimator_
print("Best Isolation Forest Parameters:", iso_grid_search.best_params_)

# Save the parameters
parametres = []
parametres.append(rf_grid_search.best_params_)
parametres.append(xgb_grid_search.best_params_)
parametres.append(svm_grid_search.best_params_)
parametres.append(iso_grid_search.best_params_)
for line in parametres:
    print(line)
    print("\n")