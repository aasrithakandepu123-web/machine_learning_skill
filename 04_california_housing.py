from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()

X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)

# Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

# Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# Predictions
linear_pred = linear.predict(X_test)
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)

# Results
print("Linear Regression")
print("MSE:", mean_squared_error(y_test, linear_pred))
print("R2:", r2_score(y_test, linear_pred))

print("\nRidge Regression")
print("MSE:", mean_squared_error(y_test, ridge_pred))
print("R2:", r2_score(y_test, ridge_pred))

print("\nLasso Regression")
print("MSE:", mean_squared_error(y_test, lasso_pred))
print("R2:", r2_score(y_test, lasso_pred))