import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Display data
print(df.head())

# Basic information
print("\nShape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

# Survival count
sns.countplot(x="survived", data=df)
plt.title("Titanic Survival Count")
plt.show()

# Survival by gender
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival by Gender")
plt.show()

# Age distribution
sns.histplot(df["age"].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation heatmap
numeric = df.select_dtypes(include="number")

sns.heatmap(numeric.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()