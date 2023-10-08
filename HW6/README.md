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

---

## Newton-Raphson Method for Logistic Regression

The Newton-Raphson method is an iterative optimization algorithm used to find the roots of a real-valued function. In the context of logistic regression, it's employed to maximize the log-likelihood function, or equivalently, minimize the logistic loss.

### Mathematical Formulation

1. **Logistic Function**:
Given an input vector \( x \) and weights vector \( \theta \), the probability \( p \) is given by:

\[
p = \frac{1}{{1 + e^{-\theta^T x}}}
\]

2. **Log-Likelihood Function**:
For logistic regression, the log-likelihood \( L(\theta) \) over a dataset with \( n \) samples is:

\[
L(\theta) = \sum_{i=1}^{n} [ y^{(i)} \log(p^{(i)}) + (1 - y^{(i)}) \log(1 - p^{(i)}) ]
\]

Where \( p^{(i)} \) is the predicted probability for the \( i^{th} \) sample and \( y^{(i)} \) is the actual label.

3. **Gradient and Hessian**:
To apply the Newton-Raphson method, we need the first and second derivatives of \( L(\theta) \). The gradient \( g \) and Hessian \( H \) are:

\[
g = \nabla L(\theta) = X^T(y - p)
\]

\[
H = -X^T W X
\]

Where \( X \) is the design matrix, \( p \) is the vector of predicted probabilities for all samples, and \( W \) is a diagonal matrix with \( i^{th} \) diagonal element as \( p^{(i)}(1 - p^{(i)}) \).

4. **Update Rule**:
The Newton-Raphson update rule is:

\[
\theta = \theta - H^{-1} g
\]

This rule is applied iteratively until convergence.

### Advantages of Newton-Raphson:

- **Convergence**: Under certain conditions, the Newton-Raphson method can converge faster than traditional gradient descent.
- **Explicit Use of Second-Order Information**: By incorporating the Hessian matrix, the method leverages second-order information about the curvature of the log-likelihood function.

### Limitations:

- **Computational Cost**: Inverting the Hessian matrix at each step can be computationally expensive, especially when dealing with high-dimensional data.
- **Saddle Points**: The method can be sensitive to saddle points where the curvature can be misleading.

---

You can incorporate this markdown explanation into your README or other documentation to provide an overview of the Newton-Raphson method in the context of logistic regression.
