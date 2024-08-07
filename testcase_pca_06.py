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

# Determine the number of principal components to retain
pca = PCA()
pca.fit(X_standardized)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components = len(cumulative_variance_ratio[cumulative_variance_ratio <= 0.95])
print(f"Number of principal components to retain: {n_components}")

# Perform PCA with the selected number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_standardized)

# Analyze loadings of each principal component
loadings = pca.components_

# Get the absolute values of loadings for each feature
abs_loadings = np.abs(loadings)

# Determine the most important feature for each principal component
most_important_features = abs_loadings.argmax(axis=1)
selected_features = feature_names[most_important_features]

print("Selected features for each principal component:")
for i, feature_index in enumerate(most_important_features):
    print(f"Principal Component {i+1}: {feature_names[feature_index]}")

# Create the scree plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot bars
explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
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

# Annotate feature names
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height / 2, selected_features[i], ha='center', va='bottom', rotation=90, color='black', fontsize=9)

plt.grid(True)

# Save scree plot data to Excel
scree_data = pd.DataFrame({'Principal Component': range(1, len(explained_variance) + 1), 'Explained Variance (%)': explained_variance})
scree_file_path = 'datasets/scree_plot_data.xlsx'  # Replace with your desired file path
scree_data.to_excel(scree_file_path, index=False, sheet_name='Scree Plot Data')

print(f"Scree plot data saved to {scree_file_path}")

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
loadings_file_path = 'datasets/loadings_data.xlsx'  # Replace with your desired file path
loadings_df.to_excel(loadings_file_path, sheet_name='Loadings')

print(f"PCA loadings saved to {loadings_file_path}")
