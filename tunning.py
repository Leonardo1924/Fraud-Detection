import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
    'n_estimators': [25,50,100,150],
    'max_depth': [3, 6, 9],
    'max_features': ['None', 'sqrt', 'log2'],
    'max_leaf_nodes': [3,6,9],
}
rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=666),
    param_distributions=rf_param_grid,
    scoring='roc_auc',
    cv=cv_strategy,
    verbose=3,
    n_jobs=-1,
    n_iter=5,
)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
print("Best Random Forest Parameters:", rf_grid_search.best_params_)

# XGBoost Tuning
print("Tuning XGBoost...")
xgb_param_grid = {
    'max_depth': [ 3, 4, 5, 6, 8, 10, 12, 15],
    'learning_rate': [0.05,0.10,0.15,0.20,0.25,0.30],
    'min_child_weight': [ 1, 3, 5, 7 ],
    'colsample_bytree': [ 0.3, 0.4, 0.5 , 0.7 ],
    'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]
}
xgb_grid_search = GridSearchCV(
    XGBClassifier(random_state=666),
    param_distributions=xgb_param_grid,
    scoring='roc_auc',
    cv=cv_strategy,
    verbose=3,
    n_jobs=-1,
    n_iter=5,
)
xgb_grid_search.fit(X_train, y_train)
best_xgb_model = xgb_grid_search.best_estimator_
print("Best XGBoost Parameters:", xgb_grid_search.best_params_)


# SVM Tuning
print("Tuning SVM...")
svm_param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf'],
}
X_train_small = X_train.sample(frac=0.2, random_state=666)
y_train_small = y_train.loc[X_train_small.index]
svm_grid_search = GridSearchCV(
    SVC(probability=True, random_state=666),  # SVM Model
    param_grid=svm_param_grid,               # Parameter grid
    scoring='roc_auc',                       # Scoring metric
    cv=cv_strategy,                                    # 3-fold cross-validation for efficiency
    verbose=3,
    n_jobs=-1                                # Use all available CPU cores
)
svm_grid_search.fit(X_train_small, y_train_small)
best_svm_model = svm_grid_search.best_estimator_
print("Best SVM Parameters:", svm_grid_search.best_params_)

# Isolation Forest Tuning
print("Tuning Isolation Forest...")
iso_param_grid = {
    'n_estimators': [100, 800, 5],
    'max_samples': [100, 500, 5],
    'contamination': [0.1, 0.2, 0.3, 0.4, 0.5],
}
iso_grid_search = GridSearchCV(
    IsolationForest(random_state=666),
    param_distributions=iso_param_grid,
    scoring='roc_auc',
    cv=cv_strategy,
    verbose=2,
    n_jobs=-1,
    n_iter=5,
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
    print("\n")
    print(line)