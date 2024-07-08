import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data from Excel file
excel_file = 'datasets/processed_dataset.xlsx'
df = pd.read_excel(excel_file)

# Columns to visualize
columns = ['Age', 'Siblings']

# Function to detect outliers using the IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ~((df[column] < lower_bound) | (df[column] > upper_bound))

# Remove outliers from the DataFrame
for column in columns:
    outlier_mask = detect_outliers(df, column)
    df = df[outlier_mask]

# Visualization for each column after removing outliers
for column in columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[column])
    plt.title(f'{column} - Outliers Removed')
    plt.show()
