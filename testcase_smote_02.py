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

# Optionally round the resampled features if they are expected to be integers
# Uncomment if needed
# X_resampled_rounded = np.round(X_resampled).astype(int)
X_resampled_rounded = X_resampled  # Use directly if not rounding

# Compute counts after SMOTE
after_counts = pd.Series(y_resampled).value_counts()
print("\nCounts after SMOTE:")
print(after_counts)

# Save the balanced data to an Excel file
balanced_data = pd.concat([pd.DataFrame(X_resampled_rounded, columns=features.columns), pd.Series(y_resampled, name=target.name)], axis=1)
output_excel_file = "datasets/balanced_data_2.xlsx"
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
    sns.histplot(X_resampled_rounded[:, i], kde=True)
    plt.title(f"{column} After SMOTE")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.show()
