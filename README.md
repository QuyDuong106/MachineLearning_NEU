# MachineLearning_NEU
This repo will store the Homework and PDF file about the Machine Learning class in the major of  DATA SCIENCE IN ECONOMICS AND BUSINESS. 
Logistic Regression is s often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. 


---

# Logistic Regression Optimization: Gradient Descent vs. Newton-Raphson

Logistic regression is a statistical method used for modeling binary outcomes. The goal is to find the best-fitting line (in a 2D space) or plane (in higher-dimensional spaces) that separates the data into two categories. The optimization of logistic regression can be achieved through different algorithms, two of which are the Gradient Descent and the Newton-Raphson methods.

## 1. Gradient Descent

### Overview
Gradient Descent is an iterative optimization algorithm used to minimize a function. In the context of logistic regression, this function is the log-likelihood.

### How It Works
1. Initialize weights (coefficients) randomly or set to zeros.
2. Calculate the gradient of the log-likelihood with respect to each weight.
3. Update the weights in the direction of the negative gradient.
4. Repeat steps 2-3 until convergence (i.e., when the change in the weights becomes very small) or until a set number of iterations.

### Pros & Cons
**Pros:**
- Simple and easy to implement.
- Can be used for other types of regression and neural networks.

**Cons:**
- Can converge slowly, especially if the learning rate is not set properly.
- Sensitive to the choice of the learning rate.

## 2. Newton-Raphson

### Overview
The Newton-Raphson method is an iterative numerical technique used to find the roots of a real-valued function. In logistic regression, it's used to maximize the log-likelihood function.

### How It Works
1. Initialize weights (coefficients) randomly or set to zeros.
2. Calculate the gradient (first derivative) of the log-likelihood with respect to each weight.
3. Calculate the Hessian matrix, which is the matrix of the second derivatives.
4. Update the weights using the inverse of the Hessian matrix multiplied by the gradient.
5. Repeat steps 2-4 until convergence (when the gradient is close to zero) or until a set number of iterations.

### Pros & Cons
**Pros:**
- Converges faster than gradient descent for logistic regression as it takes into account the curvature of the log-likelihood.
- Usually requires fewer iterations to converge.

**Cons:**
- Computationally expensive, especially when dealing with large datasets, as it requires the calculation of the inverse of the Hessian matrix.
- Can be sensitive to the initial choice of weights.

## Conclusion

Both Gradient Descent and Newton-Raphson are powerful optimization methods used in logistic regression. The choice between them depends on the specific problem, the size of the dataset, and computational resources. While Gradient Descent is more general and can be applied to various problems, Newton-Raphson can provide faster convergence for logistic regression at the cost of increased computational complexity.

---

This README provides a basic understanding of the two methods. Depending on the target audience, you might want to provide more detailed mathematical explanations or code examples.
