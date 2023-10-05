
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from logistic_regression_gd import LogisticRegression as LogisticRegressionGD
from logistic_regression_newton import LogisticRegressionNewton

# Load datasets
train_data = pd.read_csv('/mnt/data/ds1_train.csv')
test_data = pd.read_csv('/mnt/data/ds1_valid.csv')

X_train = train_data[['x_1', 'x_2']].values
y_train = train_data['y'].values
X_test = test_data[['x_1', 'x_2']].values
y_test = test_data['y'].values

# GD Model
model_gd = LogisticRegressionGD()
model_gd.fit(X_train, y_train)
y_pred_gd = model_gd.predict(X_test)

# Newton Model
model_newton = LogisticRegressionNewton()
model_newton.fit(X_train, y_train)
y_pred_newton = model_newton.predict(X_test)

# Compute accuracies
accuracy_gd = np.mean(y_pred_gd == y_test)
accuracy_newton = np.mean(y_pred_newton == y_test)

# Save accuracies to JSON
accuracy_data = {
    "accuracy_gd": accuracy_gd,
    "accuracy_newton": accuracy_newton
}
with open("/mnt/data/accuracy.json", "w") as file:
    json.dump(accuracy_data, file)

# Visualization for GD
plt.figure(figsize=(10, 6))
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', label='Class 1')
# Assuming decision boundary plotting is similar to the Newton method shown earlier
x1_values = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
x2_values_gd = # ... (code to compute decision boundary for GD)
plt.plot(x1_values, x2_values_gd, color='green', label='Decision Boundary GD')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('Decision Boundary of Logistic Regression using GD')
plt.grid(True)
plt.savefig("/mnt/data/decision_boundary_gd.png")

# Visualization for Newton
plt.figure(figsize=(10, 6))
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', label='Class 1')
x2_values_newton = # ... (code to compute decision boundary for Newton)
plt.plot(x1_values, x2_values_newton, color='green', label='Decision Boundary Newton')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('Decision Boundary of Logistic Regression using Newton')
plt.grid(True)
plt.savefig("/mnt/data/decision_boundary_newton.png")
