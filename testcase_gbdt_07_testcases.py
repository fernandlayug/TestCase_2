import pandas as pd
import joblib
import numpy as np

# Load the scalers, encoder, and model
scaler = joblib.load('model/scaler_7.pkl')
encoder = joblib.load('model/encoder_7.pkl')
best_xgb = joblib.load('model/best_xgb_model_17.pkl')

# Load the dataset
data = pd.read_excel('datasets/selected_data_top_contributing_features_1.xlsx')

# Separate features
X = data.drop(columns=['Completed'])

# Identify ordinal and non-ordinal features
ordinal_features = ['PerformanceScale', 'TravelTime', 'DaysAvailable', 'DistanceHomeSchool']  # replace with actual column names
non_ordinal_features = [col for col in X.columns if col not in ordinal_features]

# Preprocess the features
X_scaled_non_ordinal = scaler.transform(X[non_ordinal_features])
X_encoded_ordinal = encoder.transform(X[ordinal_features])

# Combine scaled non-ordinal and encoded ordinal features
X_processed = np.hstack((X_scaled_non_ordinal, X_encoded_ordinal))

# Predict probabilities
y_pred_prob = best_xgb.predict_proba(X_processed)[:, 1]

# Calculate retention and dropout percentages
data['Retention Percentage'] = y_pred_prob * 100
data['Dropout Percentage'] = (1 - y_pred_prob) * 100

# Save the updated dataset to a new Excel file
output_file = 'datasets/selected_data_with_retention_dropout.xlsx'
data.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
