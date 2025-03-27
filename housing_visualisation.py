import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "/Users/harounkheroufi/Downloads/housing.csv" 
df = pd.read_csv(file_path)

# Set the style for the plots
sns.set_style("whitegrid")

# Create subplots for histogram and scatter plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of house prices
sns.histplot(df["median_house_value"], bins=30, kde=True, ax=axes[0], color="blue")
axes[0].set_title("Histogram of House Prices")
axes[0].set_xlabel("Median House Value")
axes[0].set_ylabel("Frequency")

# Scatter plot of house prices vs. median income
sns.scatterplot(x=df["median_income"], y=df["median_house_value"], alpha=0.5, ax=axes[1], color="orange")
axes[1].set_title("House Prices vs. Median Income")
axes[1].set_xlabel("Median Income")
axes[1].set_ylabel("Median House Value")

# Show the plots
plt.tight_layout()
plt.show()

# Compute the correlation matrix using NumPy
corr_matrix = np.corrcoef(df.select_dtypes(include=[np.number]).T)
columns = df.select_dtypes(include=[np.number]).columns

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, xticklabels=columns, yticklabels=columns)

# Set title
plt.title("Correlation Heatmap")

# Show the plot
plt.show()
