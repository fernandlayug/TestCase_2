import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data from an Excel file
file_path = 'datasets/selected_data_2.xlsx'  # Replace with your file path
sheet_name = 'Sheet1'  # Replace with your sheet name if necessary
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Display the first few rows of the dataset (optional)
print(df.head())

# Separate features and target
X = df.drop(columns=['Completed']).values
y = df['Completed'].values
feature_names = df.drop(columns=['Completed']).columns

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage

# Create the scree plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot bars
bars = ax1.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', color='b')

# Plot line for explained variance
ax1.plot(range(1, len(explained_variance) + 1), explained_variance, 'ko-', markersize=8)
ax1.set_xlabel('Principal Components')
ax1.set_ylabel('Percentage of Explained Variance')
ax1.set_title('Scree Plot with Feature Names')
ax1.set_xticks(range(1, len(explained_variance) + 1))

# Annotate the explained variance percentage on the plot
for i, v in enumerate(explained_variance):
    ax1.text(i + 1, v + 1.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

# Include feature names
# Get the absolute value of loadings (coefficients of original variables in each PC)
loadings = np.abs(pca.components_)
# Normalize the loadings to sum to 1 for each PC to get their contribution percentages
loadings_normalized = loadings / loadings.sum(axis=1)[:, np.newaxis] * 100

# For each principal component, find the feature with the maximum contribution
max_contributing_features = feature_names[np.argmax(loadings_normalized, axis=1)]

# Annotate feature names
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height / 2, max_contributing_features[i], ha='center', va='bottom', rotation=90, color='black', fontsize=9)

plt.grid(True)
plt.show()

# Optional: Scatter plot of first two principal components colored by target
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Completed', data=pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]).assign(Completed=y), palette='Set1')
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Completed')
plt.show()

# Save loadings to Excel
loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=feature_names)
loadings_file_path = 'datasets/pca_data_2.xlsx'  # Replace with your desired file path
loadings_df.to_excel(loadings_file_path, sheet_name='Loadings')

print(f"PCA loadings saved to {loadings_file_path}")