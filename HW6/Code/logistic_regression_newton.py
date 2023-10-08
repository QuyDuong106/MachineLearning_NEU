import numpy as np

class LogisticRegressionNewton:
    def __init__(self, max_iter=10, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.theta = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        epsilon = 1e-5
        cost = -1/m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost

    def compute_gradient(self, X, y):
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        gradient = 1/m * X.T @ (h - y)
        return gradient

    def compute_hessian(self, X):
        m = X.shape[0]
        h = self.sigmoid(X @ self.theta)
        diag_weights = np.diag(h * (1 - h))
        hessian = 1/m * X.T @ diag_weights @ X
        return hessian

    def fit(self, X, y):
        # Add a bias term to the data
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Initialize theta if it has not been initialized
        if self.theta is None:
            self.theta = np.zeros(X.shape[1])

        for i in range(self.max_iter):
            gradient = self.compute_gradient(X, y)
            hessian = self.compute_hessian(X)
            
            # Update the theta values using the Newton-Raphson method
            self.theta = self.theta - np.linalg.inv(hessian) @ gradient

            # Check for convergence
            if np.linalg.norm(gradient) < self.tol:
                break

    def predict_prob(self, X):
        # Add a bias term to the data
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.sigmoid(X @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)
    
# Prepare the data
X_train_N = train[['x_1', 'x_2']].values
y_train_N = train['y'].values

# Initialize and train the model
model = LogisticRegressionNewton()
model.fit(X_train, y_train)


