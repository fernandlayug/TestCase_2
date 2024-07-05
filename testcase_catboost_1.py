import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import joblib
from catboost import CatBoostClassifier
import numpy as np

# Load your dataset from Excel
data = pd.read_excel('datasets/selected_data_top_contributing_features_1.xlsx')

# Separate features and target variable
X = data.drop(columns=['Completed'])  # Features
y = data['Completed']  # Target variable

# Define which features are ordinal
ordinal_features = ['PerformanceScale', 'TravelTime', 'DaysAvailable', 'DistanceHomeSchool']  # replace with actual column names
non_ordinal_features = [col for col in X.columns if col not in ordinal_features]

# Keep track of the feature names
feature_names = X.columns.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preparation: Scale the non-ordinal features using StandardScaler
scaler = StandardScaler()
X_train_scaled_non_ordinal = scaler.fit_transform(X_train[non_ordinal_features])
X_test_scaled_non_ordinal = scaler.transform(X_test[non_ordinal_features])

# Combine scaled non-ordinal and ordinal features
X_train_combined = np.hstack((X_train_scaled_non_ordinal, X_train[ordinal_features]))
X_test_combined = np.hstack((X_test_scaled_non_ordinal, X_test[ordinal_features]))

# Save the scaler to .pkl files
joblib.dump(scaler, 'model/catboost_scaler_1.pkl')

# Define the indices of categorical features for CatBoost
cat_features_indices = list(range(len(non_ordinal_features), len(non_ordinal_features) + len(ordinal_features)))

# Define the CatBoost Classifier
catboost_clf = CatBoostClassifier(random_state=42, verbose=0)

# Define the hyperparameters grid for GridSearchCV
param_grid = {
    'iterations': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 4, 5, 6],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 50, 100],
    'bagging_temperature': [0.5, 1, 2, 3]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=catboost_clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc', verbose=1)
grid_search.fit(X_train_combined, y_train, cat_features=cat_features_indices)

# Get the best parameters
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

# Use the best parameters to train the model
best_catboost = CatBoostClassifier(**best_params, random_state=42, verbose=0)
best_catboost.fit(X_train_combined, y_train, cat_features=cat_features_indices)

# Save the trained model to a .pkl file
joblib.dump(best_catboost, 'model/best_catboost_model_1.pkl')

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_catboost, X_train_combined, y_train, cv=kf, scoring='roc_auc')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Evaluation on the testing set
y_pred = best_catboost.predict(X_test_combined)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC
y_pred_prob = best_catboost.predict_proba(X_test_combined)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC AUC Score:", roc_auc)

# Feature importance
feature_importances = best_catboost.get_feature_importance()
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# Visualize feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()

# Visualize cross-validation scores
plt.figure(figsize=(8, 6))
sns.barplot(x=[f"Fold {i+1}" for i in range(len(cv_scores))], y=cv_scores)
plt.xlabel('Fold')
plt.ylabel('ROC AUC')
plt.title('Cross-Validation Scores')
plt.ylim(0.8, 1.0)
plt.show()

# Visualize confusion matrix for testing set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Testing Set')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
