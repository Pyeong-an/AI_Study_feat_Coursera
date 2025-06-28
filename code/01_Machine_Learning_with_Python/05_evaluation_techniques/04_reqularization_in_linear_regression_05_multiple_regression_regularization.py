import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression

def regression_results(y_true, y_pred, regr_type):
    # Regression metrics
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('explained_variance: ', round(ev, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mae, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))
    print()


X, y, ideal_coef = make_regression(n_samples=100, n_features=100, n_informative=10, noise=10, random_state=42, coef=True)

# Get the ideal predictions based on the informative coefficients used in the regression model
ideal_predictions = X @ ideal_coef

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X, y, ideal_predictions, test_size=0.3, random_state=42)

lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()

# Fit the models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)

# Predict on the test set
y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

regression_results(y_test, y_pred_linear, 'Ordinary')
regression_results(y_test, y_pred_ridge, 'Ridge')
regression_results(y_test, y_pred_lasso, 'Lasso')

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

axes[0, 0].scatter(y_test, y_pred_linear, color="red", label="Linear")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 0].set_title("Linear Regression")
axes[0, 0].set_xlabel("Actual", )
axes[0, 0].set_ylabel("Predicted", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )

axes[0, 1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 1].set_title("Ridge Regression", )
axes[0, 1].set_xlabel("Actual", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )

# Line plots for predictions compared to actual and ideal predictions
axes[1, 0].plot(y_test, label="Actual", lw=2)
axes[1, 0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
axes[1, 0].set_title("Linear vs Ideal", )
axes[1, 0].legend()

axes[1, 1].plot(y_test, label="Actual", lw=2)
# axes[1,1].plot(ideal_test, '--', label="Ideal", lw=2, color="purple")
axes[1, 1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1, 1].set_title("Ridge vs Ideal", )
axes[1, 1].legend()

axes[1, 2].plot(y_test, label="Actual", lw=2)
axes[1, 2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1, 2].set_title("Lasso vs Ideal", )
axes[1, 2].legend()

plt.tight_layout()
plt.show()