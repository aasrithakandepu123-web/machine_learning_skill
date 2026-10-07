import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load wine dataset
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    sep=";"
)

print(df.head())

# Convert quality into classes
# 0 = Low
# 1 = High
df["quality_class"] = (df["quality"] >= 6).astype(int)

# Features and target
X = df.drop(["quality", "quality_class"], axis=1)
y = df["quality_class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("\nDecision Tree Accuracy:",
      accuracy_score(y_test, tree_pred))


# Random Forest
forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest.fit(X_train, y_train)

forest_pred = forest.predict(X_test)

print("\nRandom Forest Accuracy:",
      accuracy_score(y_test, forest_pred))

print("\nRandom Forest Report:")
print(classification_report(y_test, forest_pred))


# Feature importance
importance = pd.Series(
    forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importance.plot(kind="bar")

plt.title("Wine Quality Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()