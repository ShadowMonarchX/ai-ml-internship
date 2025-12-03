# ⚙️ Support Vector Machines (SVM) — Full Theoretical + Mathematical Notes

---

## Overview

This document provides a detailed theoretical and mathematical explanation of **Support Vector Machines (SVM)**, a powerful supervised learning model primarily used for classification, but adaptable for regression. The core idea is to find an optimal separating hyperplane that maximizes the margin between different classes.

---

## 🧭 Table of Contents

* [1. What is an SVM?](#1-what-is-an-svm)
* [2. The Optimal Hyperplane](#2-the-optimal-hyperplane)
* [3. Mathematical Formulation (Linear SVM)](#3-mathematical-formulation-linear-svm)
* [4. The Margin and Support Vectors](#4-the-margin-and-support-vectors)
* [5. Soft Margin SVM (Handling Non-Separable Data)](#5-soft-margin-svm-handling-non-separable-data)
* [6. Non-Linear SVM and The Kernel Trick](#6-non-linear-svm-and-the-kernel-trick)
* [7. Common Kernel Functions](#7-common-kernel-functions)
* [8. Advantages and Disadvantages](#8-advantages-and-disadvantages)

---

## 1. What is an SVM?

Support Vector Machines are discriminative classifiers defined by a separating hyperplane. Given labeled training data, the algorithm outputs an optimal hyperplane that categorizes new examples.

* **Objective:** Find the hyperplane with the **maximum margin** between the nearest training data points of any class.
* **Support Vectors:** The data points closest to the hyperplane are called support vectors. They are the critical elements of the dataset, as only they influence the position and orientation of the hyperplane.



---

## 2. The Optimal Hyperplane

For a linear classification problem, multiple hyperplanes can separate the data. The SVM chooses the one that maximizes the margin.

### The Separating Hyperplane

A hyperplane in an $n$-dimensional space can be defined by the equation:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

Where:
* $\mathbf{w}$ is the weight vector (normal vector to the hyperplane).
* $\mathbf{x}$ is the input feature vector.
* $b$ is the bias term (or intercept).

---

## 3. Mathematical Formulation (Linear SVM)

The goal is to find the $\mathbf{w}$ and $b$ that maximizes the margin.

### 3.1 Classification Rule

For any data point $\mathbf{x}_i$, the classification decision is based on the sign of the expression:

$$
y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1
$$

This is the fundamental constraint for **Hard Margin SVM** (assuming the data is perfectly separable). It ensures that every point is correctly classified and lies outside the margin boundaries.

### 3.2 Maximizing the Margin

The margin is the distance between the two margin hyperplanes (where the support vectors lie):

$$\mathbf{w}^T \mathbf{x} + b = 1 \quad \text{and} \quad \mathbf{w}^T \mathbf{x} + b = -1$$

The width of this margin is $\frac{2}{\lVert \mathbf{w} \rVert}$.

Maximizing the margin is equivalent to **minimizing the magnitude of the weight vector $\mathbf{w}$**.

### 3.3 The Optimization Problem (Primal Form)

The goal is to minimize $\frac{1}{2} \lVert \mathbf{w} \rVert^2$ subject to the constraint $y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1$ for all data points $i$.

$$\begin{aligned}
\text{Minimize: } \quad & \frac{1}{2} \lVert \mathbf{w} \rVert^2 \\
\text{Subject to: } \quad & y_i (\mathbf{w}^T \mathbf{x}_i + b) - 1 \ge 0, \quad \text{for } i=1, \dots, m
\end{aligned}$$

This is a convex optimization problem that can be solved using Lagrangian multipliers (the **Dual Form**).

---

## 4. The Margin and Support Vectors

The support vectors are the data points $\mathbf{x}_i$ for which the constraint is exactly satisfied (i.e., $y_i (\mathbf{w}^T \mathbf{x}_i + b) = 1$).

* Only the support vectors, through their associated Lagrangian multipliers $\alpha_i > 0$, influence the solution $\mathbf{w}$ and $b$. All other points have $\alpha_i = 0$ and can be removed without changing the hyperplane.

---

## 5. Soft Margin SVM (Handling Non-Separable Data)

Real-world data is rarely perfectly linearly separable. The **Soft Margin SVM** introduces **slack variables ($\xi_i$)** to allow some misclassification or margin violation.

### 5.1 Slack Variables ($\xi_i$)

* $\xi_i \ge 0$: Measure the degree of misclassification or violation of the margin constraint for the $i$-th point.
* If $\xi_i = 0$: Point is correctly classified and outside the margin.
* If $0 < \xi_i < 1$: Point is correctly classified but inside the margin.
* If $\xi_i \ge 1$: Point is misclassified.

### 5.2 Soft Margin Optimization

The optimization goal is modified to minimize the margin *and* the total amount of margin violation:

$$\begin{aligned}
\text{Minimize: } \quad & \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{m} \xi_i \\
\text{Subject to: } \quad & y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i \\
& \xi_i \ge 0, \quad \text{for } i=1, \dots, m
\end{aligned}$$

* **C (Regularization Parameter):** This hyperparameter controls the trade-off between maximizing the margin (low $\frac{1}{2} \lVert \mathbf{w} \rVert^2$) and minimizing the classification error (low $\sum \xi_i$).
    * **Small C:** Favors a wider margin, tolerating more misclassifications (higher bias, lower variance).
    * **Large C:** Favors correct classification of all training data, potentially leading to a narrower margin and overfitting (lower bias, higher variance).

---

## 6. Non-Linear SVM and The Kernel Trick

When data is not linearly separable in the original input space, the **Kernel Trick** is used to implicitly map the data into a higher-dimensional feature space where it may become linearly separable.

### The Mapping Function ($\phi$)

We define a transformation $\phi(\mathbf{x})$ that maps the input vector $\mathbf{x}$ to a higher-dimensional space:

$$
\mathbf{w}^T \mathbf{\phi}(\mathbf{x}) + b = 0
$$

The problem with this direct approach is that $\phi(\mathbf{x})$ can be computationally expensive to calculate, especially for very high dimensions.

### The Kernel Trick

The Kernel Trick avoids the explicit calculation of the transformation $\phi(\mathbf{x})$. Instead, it uses a **Kernel Function ($K$)** that calculates the dot product of the two transformed vectors in the high-dimensional space:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{\phi}(\mathbf{x}_i)^T \mathbf{\phi}(\mathbf{x}_j)
$$

By replacing the dot product $\mathbf{x}_i^T \mathbf{x}_j$ in the Dual Form optimization problem with the kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$, the SVM can solve non-linear classification problems efficiently.



---

## 7. Common Kernel Functions

The choice of kernel function depends on the data structure.

1.  **Linear Kernel:**
    $$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$
    (Used for linear separation, equivalent to the Hard Margin SVM).

2.  **Polynomial Kernel:**
    $$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$
    * $d$: degree of the polynomial.
    * $\gamma$: slope.
    * $r$: constant term.

3.  **Radial Basis Function (RBF) Kernel (Gaussian Kernel):**
    $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2)$$
    * $\gamma$: controls the influence of a single training example. This is the most popular kernel, acting as a measure of similarity (distance) between data points.

---

## 8. Advantages and Disadvantages

| Advantages | Disadvantages |
| :--- | :--- |
| **Effective in High Dimensions:** Performs well even when the number of features is greater than the number of samples. | **Poor Performance on Large Datasets:** Training time increases rapidly with the size of the training set ($O(n^2)$ or $O(n^3)$). |
| **Memory Efficient:** Only uses a subset of the training points (support vectors) in the decision function. | **Sensitive to Outliers:** The margin is highly influenced by support vectors, so outliers can drastically change the optimal hyperplane. |
| **Versatile:** Can use different kernel functions to handle complex, non-linear classification problems. | **Lack of Probability Estimates:** SVMs primarily provide classification boundaries, and estimating probabilities is computationally expensive and done separately. |