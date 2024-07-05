import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import openpyxl
import numpy as np
import shap
import time

# Load your dataset from Excel (replace 'selected_data_2.xlsx' with your Excel file)
data = pd.read_excel('datasets/selected_data_top_contributing_features_1.xlsx')

# Separate features and target variable
X = data.drop(columns=['Completed'])  # Features
y = data['Completed']  # Target variable

# Define which features are ordinal
ordinal_features = ['PerformanceScale', 'TravelTime', 'DaysAvailable', 'DistanceHomeSchool']  # replace with actual column names
non_ordinal_features = [col for col in X.columns if col not in ordinal_features]

# Keep track of the feature names
feature_names = X.columns.tolist()

# Split the data into training and testing sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preparation: Scale the non-ordinal features using StandardScaler
scaler = StandardScaler()
X_train_scaled_non_ordinal = scaler.fit_transform(X_train[non_ordinal_features])
X_test_scaled_non_ordinal = scaler.transform(X_test[non_ordinal_features])

# Encode the ordinal features
encoder = OrdinalEncoder()
X_train_encoded_ordinal = encoder.fit_transform(X_train[ordinal_features])
X_test_encoded_ordinal = encoder.transform(X_test[ordinal_features])

# Combine scaled non-ordinal and encoded ordinal features
X_train_processed = np.hstack((X_train_scaled_non_ordinal, X_train_encoded_ordinal))
X_test_processed = np.hstack((X_test_scaled_non_ordinal, X_test_encoded_ordinal))

# Save the scalers and encoder to .pkl files
joblib.dump(scaler, 'model/gbdt_scaler_1.pkl')
joblib.dump(encoder, 'model/gbdt_encoder_1.pkl')

# Save the scaler to an Excel file
scaler_data = pd.DataFrame(data={
    'Feature': non_ordinal_features,
    'Mean': scaler.mean_,
    'Scale': scaler.scale_
})
scaler_data.to_excel('model/gbdt_scaler_1.xlsx', index=False)

# Save the encoder to an Excel file
encoder_data = []
for feature, categories in zip(ordinal_features, encoder.categories_):
    encoder_data.append({
        'Feature': feature,
        'Categories': ', '.join(map(str, categories))
    })

encoder_df = pd.DataFrame(encoder_data)
encoder_df.to_excel('model/gbdt_encoder_1.xlsx', index=False)

# Define the Gradient Boosting Classifier
gbdt_clf = GradientBoostingClassifier()

# Define the hyperparameters grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Measure training time
start_time = time.time()

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=gbdt_clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train_processed, y_train)

# Get the best parameters
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

# Use the best parameters to train the model
best_gbdt = GradientBoostingClassifier(**best_params)
best_gbdt.fit(X_train_processed, y_train)

# Save the trained model to a .pkl file
joblib.dump(best_gbdt, 'model/best_gbdt_model_1.pkl')

# Calculate training time
training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_gbdt, X_train_processed, y_train, cv=kf, scoring='roc_auc')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Evaluation on the testing set
y_pred = best_gbdt.predict(X_test_processed)

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
y_pred_prob = best_gbdt.predict_proba(X_test_processed)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC AUC Score:", roc_auc)

# Feature importance
feature_importances = best_gbdt.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# Visualize feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()

# SHAP values analysis
explainer = shap.Explainer(best_gbdt, X_train_processed)
shap_values = explainer(X_test_processed)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# SHAP dependence plots
for feature in feature_names:
    shap.dependence_plot(feature, shap_values.values, X_test, feature_names=feature_names)

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
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Visualize the effect of hyperparameters on cross-validated performance
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=results, x='param_learning_rate', y='mean_test_score', marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Mean ROC AUC')
plt.title('Effect of Learning Rate on ROC AUC')

plt.subplot(1, 2, 2)
sns.lineplot(data=results, x='param_n_estimators', y='mean_test_score', marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean ROC AUC')
plt.title('Effect of Number of Estimators on ROC AUC')

plt.tight_layout()
plt.show()
