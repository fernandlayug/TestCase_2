import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox
import numpy as np

def process_excel_file(file_path, output_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Visualization of missing values before processing
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Before Processing')
    plt.show()

    # Handling missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Visualization of imputed values
    imputed_mask = pd.DataFrame(imputer.transform(df) != df.values, columns=df.columns)
    plt.figure(figsize=(12, 6))
    sns.heatmap(imputed_mask, cbar=False, cmap='viridis')
    plt.title('Imputed Values')
    plt.show()

    # Handling outliers
    # Example: Detecting outliers using the IQR method
    Q1 = df_imputed.quantile(0.25)
    Q3 = df_imputed.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR)))

    # Visualization of data with outliers highlighted
    for column in df_imputed.select_dtypes(include=[float, int]).columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_imputed[column])
        outliers = df_imputed[column][outlier_mask[column]]
        plt.scatter(outliers.index, outliers, color='red', zorder=10)
        plt.title(f'{column} - Outliers Highlighted')
        plt.show()

    # Handling outliers: Apply transformation and winsorization
    df_transformed = df_imputed.copy()

    # Example of transformation: Box-Cox transformation
    for column in df_transformed.select_dtypes(include=[float, int]).columns:
        if not df_transformed[column].equals(df_imputed[column]):
            transformed_data, _ = boxcox(df_transformed[column] + 1)  # Adding 1 to handle zero and negative values
            df_transformed[column] = transformed_data

    # Example of winsorization: Limit extreme values to the 95th percentile
    for column in df_transformed.select_dtypes(include=[float, int]).columns:
        percentile_95 = np.percentile(df_transformed[column], 95)
        df_transformed[column] = np.where(df_transformed[column] > percentile_95, percentile_95, df_transformed[column])

    # Save the transformed data to a new Excel file
    df_transformed.to_excel(output_path, index=False)
    print(f"Processed file saved to: {output_path}")

# Example usage
input_file = 'datasets/encoded_data_2.xlsx'  # Replace with your input file path
output_file = 'datasets/processed_dataset.xlsx'  # Replace with your desired output file path

process_excel_file(input_file, output_file)
