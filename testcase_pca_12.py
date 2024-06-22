import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data from an Excel file
file_path = 'datasets/selected_data_4.xlsx'  # Replace with your file path
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

# Determine the number of components to retain
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_) * 100
n_components_98 = np.where(cumulative_explained_variance >= 98)[0][0] + 1  # 98% variance
print(f"Number of components to retain for 98% variance: {n_components_98}")

# Use the number of components to retain 
n_components_to_retain = n_components_98

# Get the loadings (coefficients of the original features for each principal component)
loadings = pca.components_.T

# Create a DataFrame for loadings
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(loadings.shape[1])], index=feature_names)

# Identify the features that contribute the most to the retained components without redundancy
top_features = {}
used_features = set()

for i in range(n_components_to_retain):
    pc_loadings = loadings_df.iloc[:, i]
    sorted_features = pc_loadings.abs().sort_values(ascending=False)
    
    # Find the first feature that has not been used yet
    for feature in sorted_features.index:
        if feature not in used_features:
            top_features[f'PC{i+1}'] = feature
            used_features.add(feature)
            break

print("Top contributing features for each retained component without redundancy:")
for pc, feature in top_features.items():
    print(f"{pc}: {feature}")

# Save the dataset with only the top contributing features
top_contributing_features = list(used_features) + ['Completed']
df_top_features = df[top_contributing_features]

top_features_file_path = 'datasets/selected_data_top_contributing_features_1.xlsx'  # Replace with your desired file path
df_top_features.to_excel(top_features_file_path, index=False)

print(f"Dataset with top contributing features saved to {top_features_file_path}")

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
loadings_abs = np.abs(loadings)
# Normalize the loadings to sum to 1 for each PC to get their contribution percentages
loadings_normalized = loadings_abs / loadings_abs.sum(axis=0) * 100

# For each principal component, find the feature with the maximum contribution without redundancy
used_features_plot = set()

for i, bar in enumerate(bars):
    for feature in loadings_normalized[:, i].argsort()[::-1]:
        feature_name = feature_names[feature]
        if feature_name not in used_features_plot:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, 0, feature_name, ha='center', va='top', rotation=90, color='black', fontsize=9)
            used_features_plot.add(feature_name)
            break

plt.grid(True)
plt.show()

# Plot cumulative explained variance
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
plt.axhline(y=98, color='r', linestyle='-')
plt.axvline(x=n_components_98, color='r', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Explained Variance Plot')
plt.grid(True)
plt.show()

# Print the components to retain
print(f"Components to retain for 98% cumulative variance: {list(range(1, n_components_98 + 1))}")

# Optional: Scatter plot of first two principal components colored by target
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Completed', data=pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]).assign(Completed=y), palette='Set1')
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Completed')
plt.show()

# Create a heatmap of the retained components
plt.figure(figsize=(10, 8))
sns.heatmap(loadings_df.iloc[:, :n_components_to_retain], cmap='coolwarm', annot=True)
plt.title('Heatmap of PCA Loadings for Retained Components')
plt.xlabel('Principal Components')
plt.ylabel('Features')
plt.show()

# Save loadings to Excel
loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=feature_names)
loadings_file_path = 'datasets/pca_data_3.xlsx'  # Replace with your desired file path
loadings_df.to_excel(loadings_file_path, sheet_name='Loadings')

print(f"PCA loadings saved to {loadings_file_path}")

# Save dataset with selected features and target variable to an Excel file
selected_features = list(used_features) + ['Completed']
selected_data_df = df[selected_features]
selected_data_file_path = 'datasets/selected_data_with_selected_features_1.xlsx'  # Replace with your desired file path
selected_data_df.to_excel(selected_data_file_path, index=False)

print(f"Dataset with selected features saved to {selected_data_file_path}")
