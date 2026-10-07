from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load diabetes data
data = load_diabetes()

X = data.data
y_original = data.target

# Create 3 severity classes
y = np.digitize(
    y_original,
    bins=[100, 150]
)

# Classes:
# 0 = Low
# 1 = Medium
# 2 = High

print("Classes:", np.unique(y))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Multinomial Logistic Regression
model = LogisticRegression(
    multi_class="multinomial",
    max_iter=1000
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))