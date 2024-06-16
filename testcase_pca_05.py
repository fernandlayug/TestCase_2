import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Read data from Excel file into pandas DataFrame
df = pd.read_excel("datasets/selected_data_2.xlsx")

# Step 2: Exclude the target column
target_column = "Completed"
df_features = df.drop(columns=[target_column])

# Step 3: Preprocess the data if necessary
# For example, handle missing values or encode categorical variables

# Step 4: Standardize the features
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df_features)

# Step 5: Compute the covariance matrix
cov_matrix = np.cov(df_standardized, rowvar=False)

# New Step: Visualize the covariance matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=df_features.columns, yticklabels=df_features.columns)
plt.title('Covariance Matrix Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

# Step 6: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 7: Apply PCA
pca = PCA()
pca.fit(df_standardized)

# Step 8: Analyze explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Scree plot with labels and feature names
plt.figure(figsize=(12, 8))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid',
         label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot with Feature Names')
plt.legend(loc='best')

for i, (ev, cv) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio), start=1):
    plt.text(i, ev, f"{ev:.2f}", ha='center', va='bottom', fontsize=8, rotation=90)
    plt.text(i, cv, f"{cv:.2f}", ha='center', va='top', fontsize=8)
    
# Add feature names to the x-axis
feature_names = df_features.columns
plt.xticks(ticks=range(1, len(explained_variance_ratio) + 1), labels=[f"PC{i}" for i in range(1, len(explained_variance_ratio) + 1)], rotation=90)

plt.tight_layout()
plt.show()

# Determine the number of principal components to retain
n_components = len(cumulative_variance_ratio[cumulative_variance_ratio <= 0.95])

# Step 9: Transform original features into new feature space using selected principal components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(df_standardized)

# Step 10: Analyze loadings of each principal component
loadings = pca.components_

# Get the absolute values of loadings for each feature
abs_loadings = np.abs(loadings)

# Determine the most important feature for each principal component
most_important_features = abs_loadings.argmax(axis=1)

# Get the names of the selected features for each principal component
selected_features = df_features.columns[most_important_features]

print("Selected features for each principal component:")
for i, feature_index in enumerate(most_important_features):
    print(f"Principal Component {i+1}: {df_features.columns[feature_index]}")

# Create a DataFrame for loadings
loadings_df = pd.DataFrame(loadings, columns=df_features.columns)

# Plot heatmap of PCA loadings
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('PCA Loadings Heatmap')
plt.xlabel('Original Features')
plt.ylabel('Principal Components')
plt.show()

# Additional visualization: Scatter plot of the first two principal components
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], edgecolor='k', s=50, alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of First Two Principal Components')
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary of Covariance Matrix
cov_summary = {
    'Diagonal (variances)': np.diag(cov_matrix),
    'Mean of off-diagonal (covariances)': np.mean(cov_matrix - np.diag(np.diag(cov_matrix))),
    'Min covariance': np.min(cov_matrix),
    'Max covariance': np.max(cov_matrix)
}

print("\nSummary of Covariance Matrix:")
for key, value in cov_summary.items():
    print(f"{key}: {value}")

# Summary of Eigenvalues
eigenvalues_summary = {
    'Eigenvalues': eigenvalues,
    'Sum of eigenvalues': np.sum(eigenvalues),
    'Mean eigenvalue': np.mean(eigenvalues),
    'Min eigenvalue': np.min(eigenvalues),
    'Max eigenvalue': np.max(eigenvalues)
}

print("\nSummary of Eigenvalues:")
for key, value in eigenvalues_summary.items():
    print(f"{key}: {value}")
