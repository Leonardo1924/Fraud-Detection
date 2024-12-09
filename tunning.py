import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the preprocessed data
train_data = pd.read_csv('Data/train_processed.csv')
test_data = pd.read_csv('Data/test_processed.csv')

# Split the data into features and target
X_train = train_data.drop('is_fraud', axis=1)
y_train = train_data['is_fraud']
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud']

# --------------------------------#
#       Logistic Regression       #
# --------------------------------#
print("Tuning Logistic Regression...")
log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, verbose=1)
grid_log_reg.fit(X_train, y_train)

log_reg = grid_log_reg.best_estimator_

# --------------------------------#
#       KNearest Neighbors        #
# --------------------------------#
print("Tuning KNearest Neighbors...")
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params, verbose=1)
grid_knears.fit(X_train, y_train)

knears_neighbors = grid_knears.best_estimator_

#---------------------------------#
#    Support Vector Classifier    #
#---------------------------------#
print("Tuning Support Vector Classifier...")
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params, verbose=1)
grid_svc.fit(X_train, y_train)

svc = grid_svc.best_estimator_

#-------------------------------#
#    DecisionTree Classifier    #
#-------------------------------#
print("Tuning DecisionTree Classifier...")
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params , verbose=1)
grid_tree.fit(X_train, y_train)

tree_clf = grid_tree.best_estimator_

#-------------------------------#
#    Random Forest Classifier   #
#-------------------------------#
print("Tuning Random Forest Classifier...")
rf_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}

grid_rf = GridSearchCV(RandomForestClassifier(), rf_params , verbose=1)
grid_rf.fit(X_train, y_train)

# Random forest best estimator
rf_clf = grid_rf.best_estimator_

# Create a dictionary of the best estimators
best_estimators = {'Logistic Regression': log_reg, 'KNearest': knears_neighbors, 
                   'Support Vector Classifier': svc, 'Decision Tree': tree_clf, 
                   'Random Forest': rf_clf}

# Save the best estimators
joblib.dump(best_estimators, 'Models/best_estimators.pkl')
print("Best estimators saved successfully.")

joblib.dump(X_train.columns, 'Models/features.pkl')
print("Feature names saved successfully.")