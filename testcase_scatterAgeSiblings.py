import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read data from Excel
file_path = 'datasets/processed_dataset.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Step 2: Create separate scatter plots with x and y axes interchanged
plt.figure(figsize=(14, 6))

# Scatter plot with x and y axes interchanged for Age
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
plt.scatter(df.index, df['Age'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Age')
plt.xlabel('Index')
plt.ylabel('Age')

# Scatter plot with x and y axes interchanged for Siblings
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
plt.scatter(df.index, df['Siblings'], alpha=0.5, color='green')
plt.title('Scatter Plot of Siblings')
plt.xlabel('Index')
plt.ylabel('Siblings')

plt.tight_layout()  # Adjust spacing between plots
plt.show()