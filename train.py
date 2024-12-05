import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load the dataset
dataset_path = 'Data/processed_dataset.csv'
dataset = pd.read_csv(dataset_path)

X = dataset.drop(['is_fraud'], axis=1)
y = dataset['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# Train an XGBoost model
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Results:")
print(classification_report(y_test, y_pred_xgb, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

# Train a Support Vector Machine model

# Step 1: Subsample the dataset
subsample_fraction = 0.5  # Use 10% of the data
X_train_small = X_train.sample(frac=subsample_fraction, random_state=42)
y_train_small = y_train.loc[X_train_small.index]

# Step 2: Dimensionality Reduction using PCA
pca = PCA(n_components=10, random_state=42)  # Reduce to 10 components
X_train_small_pca = pca.fit_transform(X_train_small)
X_test_pca = pca.transform(X_test)

svm_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42, C=1, gamma='scale')
svm_model.fit(X_train_small_pca, y_train_small)

y_pred = svm_model.predict(X_test_pca)
y_pred_probs = svm_model.predict_proba(X_test_pca)[:, 1]

# Step 5: Evaluate the Model
print("\nSVM (RBF Kernel) Results:")
print(classification_report(y_test, y_pred, zero_division=0))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_probs))

# Check feature importances
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances.head(10))

## Save the best model and features
joblib.dump(X.columns.tolist(), 'Models/trained_features.pkl')
print("Feature list saved as 'trained_features.pkl'.")

joblib.dump(rf_model, 'Models/random_forest_model.pkl')
print("Best model saved as 'random_forest_model.pkl'")

joblib.dump(xgb_model, 'Models/xgbboost.pkl')
print("Best model saved as 'xgbboost.pkl'")
