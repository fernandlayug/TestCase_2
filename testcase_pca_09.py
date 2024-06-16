import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data from an Excel file
file_path = 'datasets/selected_data_3.xlsx'  # Replace with your file path
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

# Perform PCA to determine the number of principal components to retain
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Determine the number of principal components to retain
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

# Perform PCA with the selected number of principal components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_standardized)

# Analyze loadings of each principal component
loadings = pca.components_

# Get the absolute values of loadings for each feature
abs_loadings = np.abs(loadings)

# Identify the most significant feature for each principal component
most_significant_features = []
for i in range(n_components):
    most_significant_feature = feature_names[np.argmax(abs_loadings[i])]
    most_significant_features.append(most_significant_feature)

print("Most significant feature for each principal component:")
for i, feature in enumerate(most_significant_features):
    print(f"Principal Component {i + 1}: {feature}")

# Create a DataFrame for loadings
loadings_df = pd.DataFrame(loadings, columns=feature_names)

# Plot heatmap of PCA loadings
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('PCA Loadings Heatmap')
plt.xlabel('Original Features')
plt.ylabel('Principal Components')
plt.show()

# Create the enhanced scree plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot bars
bars = ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.7, align='center', color='b')

# Plot line for cumulative explained variance
ax1.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio * 100, 'ko-', markersize=8)
ax1.set_xlabel('Principal Components')
ax1.set_ylabel('Percentage of Explained Variance')
ax1.set_title('Scree Plot with Feature Names')
ax1.set_xticks(range(1, len(explained_variance_ratio) + 1))

# Annotate the explained variance percentage on the plot
for i, v in enumerate(cumulative_variance_ratio * 100):
    ax1.text(i + 1, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

# Annotate the most significant feature names
for i, feature in enumerate(most_significant_features):
    ax1.text(i + 1, explained_variance_ratio[i] * 100 + 0.5, feature, ha='center', va='bottom', fontsize=8, rotation=90)

plt.grid(True)
plt.show()

# Optional: Scatter plot of first two principal components colored by target
plt.figure(figsize=(10, 6))
sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], hue=y, palette='Set1')
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Completed')
plt.show()

# Save loadings to Excel
loadings_df.to_excel('datasets/loadings_data.xlsx', sheet_name='Loadings', index=False)
print("PCA loadings saved to datasets/loadings_data.xlsx")
