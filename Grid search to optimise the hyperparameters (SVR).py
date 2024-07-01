import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import validation_curve

# Load the training data and test data
df_train = pd.read_csv("216_850nm.csv")
df_test = pd.read_csv("test set_850nm.csv")

# Define the feature variables and target variable for the training data
names = ('Pitch', 'Duty', 'Height')
X_train = df_train.loc[:, names]
y_train = df_train['T']

# Define the feature variables for the test data
X_test = df_test.loc[:, names]
y_test = df_test['T']

# Create a pipeline that includes the StandardScaler for feature scaling and the SVM model for regression
reg_T = Pipeline([
    ('scl', StandardScaler()),
    ('clf', svm.SVR(kernel='rbf'))
])

# Perform parameter optimization using GridSearchCV
param_grid = {
    'clf__gamma': [0.15, 0.25, 0.35, 0.45],
    'clf__C': [3, 5, 10],
    'clf__epsilon': [0.1, 0.2, 0.5]
}

grid_search = GridSearchCV(reg_T, param_grid, scoring='neg_mean_squared_error', cv=10)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Fit the best model to the training data
best_model.fit(X_train, y_train)

# Predict the target variable for the test data
y_test_pred = best_model.predict(X_test)
print(y_test_pred)

# Calculate MSE and R-squared for performance comparison
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Performance Metrics:")
print("MSE:", mse)
print("R-squared:", r2)

# Generate validation curves for gamma
param_range = [0.15, 0.25, 0.35, 0.45]
train_scores, test_scores = validation_curve(
    best_model, X_train, y_train, param_name="clf__gamma", param_range=param_range, scoring="neg_mean_squared_error", cv=10
)

# Calculate mean and standard deviation of training and test scores
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve for gamma
plt.figure()
plt.title("Validation Curve - Gamma")
plt.xlabel("Gamma")
plt.ylabel("Mean Squared Error")
plt.ylim((0.0, 0.05))
plt.grid()

plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(param_range, train_mean, color="r", label="Training score")
plt.plot(param_range, test_mean, color="g", label="Cross-validation score")

plt.legend(loc="best")

# Generate validation curves for C
param_range = [3, 5, 10]
train_scores, test_scores = validation_curve(
    best_model, X_train, y_train, param_name="clf__C", param_range=param_range, scoring="neg_mean_squared_error", cv=10
)

# Calculate mean and standard deviation of training and test scores
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve for C
plt.figure()
plt.title("Validation Curve - C")
plt.xlabel("C")
plt.ylabel("Mean Squared Error")
plt.ylim((0.0, 0.05))
plt.grid()

plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(param_range, train_mean, color="r", label="Training score")
plt.plot(param_range, test_mean, color="g", label="Cross-validation score")

plt.legend(loc="best")

# Generate validation curves for epsilon
param_range = [0.1, 0.2, 0.5]
train_scores, test_scores = validation_curve(
    best_model, X_train, y_train, param_name="clf__epsilon", param_range=param_range, scoring="neg_mean_squared_error", cv=10
)

# Calculate mean and standard deviation of training and test scores
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve for epsilon
plt.figure()
plt.title("Validation Curve - Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Mean Squared Error")
plt.ylim((0.0, 0.2))
plt.grid()

plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(param_range, train_mean, color="r", label="Training score")
plt.plot(param_range, test_mean, color="g", label="Cross-validation score")

plt.legend(loc="best")

plt.show()
