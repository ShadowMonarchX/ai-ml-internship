# 📚 Logistic Regression — Full Theoretical + Mathematical Notes

---

## Overview

This document provides a deep dive into **Logistic Regression**, a powerful classification algorithm, covering its core theory, mathematical formulation, and optimization methods.

---

## 🧭 Table of Contents

* [1. What is Logistic Regression?](#1-what-is-logistic-regression)
* [2. The Sigmoid Function](#2-the-sigmoid-function)
* [3. Mathematical Model](#3-mathematical-model)
* [4. Decision Boundary](#4-decision-boundary)
* [5. Cost Function: Log Loss (Cross-Entropy)](#5-cost-function-log-loss-cross-entropy)
* [6. Optimization: Gradient Descent](#6-optimization-gradient-descent)
* [7. Multiclass Classification (Softmax)](#7-multiclass-classification-softmax)
* [8. Evaluation Metrics](#8-evaluation-metrics)
* [9. Assumptions & Limitations](#9-assumptions--limitations)

---

## 1. What is Logistic Regression?

Logistic Regression is a **linear model** used for **classification** tasks, unlike Linear Regression, which is used for regression.

* **Goal:** To estimate the probability that an input belongs to a certain class (e.g., probability of $y=1$ given $X$).
* **Output:** The result is always a probability $P(y|X)$, constrained between 0 and 1.
* **Classification:** The calculated probability is converted into a binary class (0 or 1) using a **threshold** (typically 0.5).

### Why not use Linear Regression for Classification?

Linear Regression outputs continuous values $(-\infty, \infty)$, which are unsuitable for probabilities $(0, 1)$. Furthermore, outliers would severely skew the linear boundary.

---

## 2. The Sigmoid Function

The core of Logistic Regression is the **Sigmoid function** (or Logistic function), $\sigma(z)$, which squashes any real-valued number into a value between 0 and 1.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$



[Image of Sigmoid function graph]


**Key Properties:**
* As $z \to \infty$, $\sigma(z) \to 1$.
* As $z \to -\infty$, $\sigma(z) \to 0$.
* If $z = 0$, $\sigma(z) = 0.5$.

---

## 3. Mathematical Model

### 3.1 Linear Combination ($z$)

First, the model calculates a linear combination of the input features and weights, similar to Linear Regression:

$$z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$$

In vector form:
$$z = \mathbf{w}^T \mathbf{x}$$

### 3.2 Hypothesis Function ($\hat{y}$)

The hypothesis function, $h_{\mathbf{w}}(\mathbf{x})$, applies the Sigmoid function to $z$ to get the estimated probability $\hat{y}$:

$$\hat{y} = h_{\mathbf{w}}(\mathbf{x}) = P(y=1 | \mathbf{x}; \mathbf{w}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x})}}$$

### 3.3 Probability Relationships

The model defines the probability of belonging to class 1 and class 0:

* $P(y=1 | \mathbf{x}; \mathbf{w}) = \hat{y}$
* $P(y=0 | \mathbf{x}; \mathbf{w}) = 1 - \hat{y}$

---

## 4. Decision Boundary

The decision boundary is the point where the model switches its prediction from 0 to 1, which happens when the estimated probability $\hat{y}$ is 0.5.

Since $\sigma(z) = 0.5$ when $z=0$:

$$\mathbf{w}^T \mathbf{x} = 0$$

* If $\mathbf{w}^T \mathbf{x} > 0$, then $\hat{y} > 0.5$, and the prediction is **Class 1**.
* If $\mathbf{w}^T \mathbf{x} < 0$, then $\hat{y} < 0.5$, and the prediction is **Class 0**.

The equation $\mathbf{w}^T \mathbf{x} = 0$ defines the hyperplane (line in 2D) that separates the two classes.



---

## 5. Cost Function: Log Loss (Cross-Entropy)

Logistic Regression uses the **Log Loss** function (or Binary Cross-Entropy) because the Sigmoid function would result in a non-convex MSE function with many local minima, making gradient descent unreliable.

The Log Loss function is convex, ensuring gradient descent converges to a single global minimum.

### 5.1 Cost for a Single Sample

The cost function $J(\mathbf{w})$ for a single training example $(\mathbf{x}^{(i)}, y^{(i)})$ is:

$$Cost(h_{\mathbf{w}}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases} -\log(h_{\mathbf{w}}(\mathbf{x}^{(i)})) & \text{if } y^{(i)} = 1 \\ -\log(1 - h_{\mathbf{w}}(\mathbf{x}^{(i)})) & \text{if } y^{(i)} = 0 \end{cases}$$

This can be written compactly:

$$\text{Cost}(\hat{y}, y) = -y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})$$

### 5.2 Total Cost Function

The total cost $J(\mathbf{w})$ is the average cost over all $m$ training examples:

$$J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\mathbf{w}}(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\mathbf{w}}(\mathbf{x}^{(i)})) \right]$$

**Goal:** Find $\mathbf{w}$ that **minimizes** $J(\mathbf{w})$.

---

## 6. Optimization: Gradient Descent

We use Gradient Descent to iteratively update the weight vector $\mathbf{w}$ to minimize the cost function $J(\mathbf{w})$.

### 6.1 Update Rule

The weights are updated simultaneously:
$$w_j := w_j - \alpha \frac{\partial J(\mathbf{w})}{\partial w_j}$$

Where $\alpha$ is the learning rate.

### 6.2 Gradient Derivation

The partial derivative of the cost function with respect to the weight $w_j$ is:

$$\frac{\partial J(\mathbf{w})}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

### 6.3 Final Update Rule (Vectorized)

The weight vector $\mathbf{w}$ is updated in a single step:

$$\mathbf{w} := \mathbf{w} - \frac{\alpha}{m} \mathbf{X}^T (\mathbf{\hat{y}} - \mathbf{y})$$

This update rule is mathematically identical to the one used in Linear Regression, but the prediction $\mathbf{\hat{y}}$ is calculated differently (using the Sigmoid function).

---

## 7. Multiclass Classification (Softmax)

When there are more than two classes (e.g., classifying images of cats, dogs, or birds), Logistic Regression is extended using the **Softmax function**, making it **Softmax Regression** (or Multinomial Logistic Regression).

### Softmax Function

The Softmax function takes a vector of $K$ scores (one for each class) and squashes them into a probability distribution, where all probabilities sum to 1.

For a data point $\mathbf{x}$, the probability of belonging to class $k$ is:

$$P(y=k | \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}$$

---

## 8. Evaluation Metrics

Because Logistic Regression is a classification model, we use classification metrics:

1.  **Accuracy:** $\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$
2.  **Precision:** $\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ (How many selected items are relevant?)
3.  **Recall:** $\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ (How many relevant items are selected?)
4.  **F1 Score:** The harmonic mean of Precision and Recall. $F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
5.  **Confusion Matrix:** A table summarizing True Positives, True Negatives, False Positives, and False Negatives.



---

## 9. Assumptions & Limitations

### Assumptions

1.  **Binary Outcome:** Assumes the dependent variable is dichotomous (for Binary Logistic Regression).
2.  **Independence of Errors:** Observations must be independent.
3.  **No Multicollinearity:** Features should not be highly correlated.
4.  **Linearity in Log-Odds:** Assumes a linear relationship between the independent variables and the **log-odds** ($\log(\frac{P}{1-P})$), not the outcome itself.

### Limitations

* **Only Linear Boundaries:** It can only model linear decision boundaries. It cannot handle complex non-linear relationships without manual feature engineering (e.g., adding polynomial features).
* **Sensitive to Outliers:** Like Linear Regression, it can be sensitive to influential outliers.