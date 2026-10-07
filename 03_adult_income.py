import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml

# Load Adult Income dataset
data = fetch_openml("adult", version=2, as_frame=True)

df = data.frame

print(df.head())
print("\nShape:", df.shape)

# Rename target
df.rename(columns={"class": "income"}, inplace=True)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Age groups
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 40, 60, 100],
    labels=["Young", "Adult", "Middle Age", "Senior"]
)

print("\nAge Groups:")
print(df["age_group"].value_counts())

# Income distribution
sns.countplot(x="income", data=df)
plt.title("Income Distribution")
plt.show()

# Education distribution
plt.figure(figsize=(10, 5))
sns.countplot(y="education", data=df)
plt.title("Education Distribution")
plt.show()

# Age distribution
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.show()