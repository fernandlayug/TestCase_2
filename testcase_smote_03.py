import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer  # or IterativeImputer
from imblearn.over_sampling import SMOTE
import numpy as np

# Read data from Excel file
excel_file = "datasets/encoded_data_1.xlsx"
data = pd.read_excel(excel_file)

# Handle missing values with KNN Imputer
imputer = KNNImputer(n_neighbors=5)  # Or use IterativeImputer for multivariate relationships
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Separate features and target variable
features = data_imputed.iloc[:, :-1]  # Assuming the last column is the target variable
target = data_imputed.iloc[:, -1]

# Compute counts before SMOTE
before_counts = target.value_counts()
print("Counts before SMOTE:")
print(before_counts)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

# Function to round and convert to nearest integer for categorical and binary features
def round_and_convert(data, original_data):
    # Creating a copy to avoid changes to the original data
    rounded_data = data.copy()
    
    for column in rounded_data.columns:
        # Check if the original data column is categorical or binary
        if original_data[column].dtype == 'category' or len(original_data[column].unique()) < 10:
            # Round and convert to integers
            rounded_data[column] = np.round(rounded_data[column]).astype(int)
            # Convert back to original categories if categorical
            if original_data[column].dtype.name == 'category':
                # Map integers back to original categories
                categories = original_data[column].cat.categories
                rounded_data[column] = rounded_data[column].map(lambda x: categories[x])
    return rounded_data

# Apply rounding and conversion
X_resampled_rounded = round_and_convert(pd.DataFrame(X_resampled, columns=features.columns), data[features.columns])

# Compute counts after SMOTE
after_counts = pd.Series(y_resampled).value_counts()
print("\nCounts after SMOTE:")
print(after_counts)

# Save the balanced data to an Excel file
balanced_data = pd.concat([X_resampled_rounded, pd.Series(y_resampled, name=target.name)], axis=1)
output_excel_file = "datasets/balanced_data_3.xlsx"
balanced_data.to_excel(output_excel_file, index=False)

# Visualization
num_features = len(features.columns)
plt.figure(figsize=(14, 8))

# Visualization before SMOTE
for i, column in enumerate(features.columns):
    plt.subplot(2, num_features, i + 1)
    sns.histplot(data[column], kde=True)
    plt.title(f"{column} Before SMOTE")
    plt.xlabel("")
    plt.ylabel("")

# Visualization after SMOTE
for i, column in enumerate(features.columns):
    plt.subplot(2, num_features, num_features + i + 1)
    sns.histplot(X_resampled_rounded[column], kde=True)
    plt.title(f"{column} After SMOTE")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.show()
