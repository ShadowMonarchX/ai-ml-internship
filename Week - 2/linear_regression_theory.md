# 📘 Linear Regression — Full Theoretical + Mathematical Notes

---

## Overview

This repository contains final study notes for **Linear Regression** (deep theoretical explanation with maths) and a ready-to-copy Python implementation you can paste into your Git repo. The README is formatted for easy reading and direct use as study material.

---

## 📚 Table of Contents

* [1. What is Linear Regression?](#1-what-is-linear-regression)
* [2. Types of Linear Regression](#2-types-of-linear-regression)
* [3. Assumptions of Linear Regression (Very Important)](#3-assumptions-of-linear-regression-very-important)
* [4. Mathematical Formulation](#4-mathematical-formulation)
* [5. Cost Function — Mean Squared Error (MSE)](#5-cost-function--mean-squared-error-mse)
* [6. Finding Best Coefficients](#6-finding-best-coefficients)
* [7. Evaluation Metrics](#7-evaluation-metrics)
* [8. Gradient Descent Variants](#8-gradient-descent-variants)
* [9. Problems with Linear Regression](#9-problems-with-linear-regression)
* [10. Regularization](#10-regularization)
* [11. Geometric & Statistical Interpretation](#11-geometric--statistical-interpretation)
* [12. Residual Analysis](#12-residual-analysis)
* [13. When to Use / Not Use](#13-when-to-use--not-use)
* [14. Summary (Revision)](#14-summary-revision)

---

## 1. What is Linear Regression?

Linear Regression is a **supervised learning algorithm** used to model the relationship between:

* **Independent variables (features)** → $X$
* **Dependent variable (target)** → $y$

**Goal:** Find a **best-fit straight line (or hyperplane)** that predicts $y$ from $X$.


[Image of Simple Linear Regression best-fit line plotted on scatter data]


---

## 2. Types of Linear Regression

### 2.1 Simple Linear Regression
* One feature:
$$\text{y} = \beta_0 + \beta_1 x + \varepsilon$$

### 2.2 Multiple Linear Regression
* Multiple features:
$$\text{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon$$

### 2.3 Polynomial Regression
* Handle non-linear relations by adding polynomial features (still linear in coefficients).

---

## 3. Assumptions of Linear Regression (Very Important)

1.  **Linearity** — relationship between features and target is linear.
2.  **Independence** — observations are independent.
3.  **Homoscedasticity** — **constant variance** of residuals.
4.  **Normality of errors** — residuals approximately normally distributed.
5.  **No multicollinearity** — features are not highly correlated.

> If these assumptions fail, statistical inference (like p-values and confidence intervals) becomes unreliable.

---

## 4. Mathematical Formulation

### **Matrix Notation**
$$\text{y} = X\beta + \varepsilon$$

**Where:**
* $X$ — feature matrix ($m \times n$)
* $\beta$ — coefficient vector
* $y$ — target vector
* $\varepsilon$ — error term (noise)

---

## 5. Cost Function — Mean Squared Error (MSE)

$$\text{J}(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

Where $\hat{y} = X\beta$. The goal is to **minimize** $J(\beta)$.

---

## 6. Finding Best Coefficients

### 6.1 Normal Equation (Closed-form)
$$\beta = (X^T X)^{-1} X^T y$$

* **Pros:** exact solution, no learning rate ($\alpha$).
* **Cons:** expensive for large $n$ (requires computing $(X^T X)^{-1}$), requires invertible $X^TX$.

### 6.2 Gradient Descent (Iterative)
**Gradient of the cost:**
$$\nabla_{\beta} J = -\frac{1}{m} X^T (y - X\beta)$$

**Update rule (vectorized):**
$$\beta := \beta + \alpha \frac{1}{m} X^T (y - X\beta)$$

* $\alpha$ is the **learning rate**. Repeat till convergence.

---

## 7. Evaluation Metrics

### R² (Coefficient of Determination)
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

where $SS_{res} = \sum (y - \hat{y})^2$ and $SS_{tot} = \sum (y - \bar{y})^2$.

### Other Metrics
* **Adjusted R²:** penalizes additional features.
* **MSE / RMSE / MAE:** common error metrics.

---

## 8. Gradient Descent Variants

1.  **Batch GD:** uses all training data per update.
2.  **Stochastic GD:** uses one sample per update.
3.  **Mini-batch GD:** uses small batches (common practice).

---

## 9. Problems with Linear Regression

1.  **Outliers** can heavily influence the fit.
2.  **Multicollinearity** $\rightarrow$ unstable coefficient estimates.
3.  **Underfitting** when relation is non-linear.

**Remedies:** robust regression, remove/transform outliers, regularization, feature engineering.

---

## 10. Regularization

Used to reduce **overfitting** and handle multicollinearity by penalizing large coefficients ($\beta$).

### Ridge (L2)
$$\text{Loss} = MSE + \lambda \sum_{j} \beta_j^2$$

### Lasso (L1)
$$\text{Loss} = MSE + \lambda \sum_{j} |\beta_j|$$

### ElasticNet
* Combination of L1 and L2 penalties.

---

## 11. Geometric & Statistical Interpretation

* **Geometric:** Find a **hyperplane** minimizing squared perpendicular distances.
* **Statistical:** Under the assumption $\varepsilon \sim N(0, \sigma^2)$, OLS estimates are equivalent to Maximum Likelihood Estimates (MLE).

**Simple 1D slope formula:**
$$\beta_1 = \frac{Cov(X,Y)}{Var(X)}, \qquad \beta_0 = \bar{y} - \beta_1 \bar{x}$$

---

## 12. Residual Analysis

**Residuals:** $e_i = y_i - \hat{y}_i$.


**Good model:** residuals are **randomly scattered** (no pattern), constant variance (Homoscedasticity), and approximately normally distributed.

* Use residual plots, Q–Q plots, and statistical tests to validate assumptions.

---

## 13. When to Use / Not Use

| Use When: | Avoid When: |
| :--- | :--- |
| **Linear** relation is known or assumed. | Complex **non-linear** patterns exist. |
| **Interpretability** of coefficients is required. | High **multicollinearity** is present. |
| Data is relatively **clean** (no severe outliers). | Prediction accuracy is the *only* goal and better non-linear models exist. |

---

## 14. Summary (Revision)
<<<<<<< HEAD

* Linear Regression predicts **continuous** outcomes.
* Minimizes **MSE** via OLS.
* Solve via **Normal Equation** or **Gradient Descent**.
* **Regularize** (Ridge/Lasso) to combat overfitting.
* **Validate assumptions** with residual analysis.
=======
>>>>>>> e1f0279 (week - 2 Day 2)

* Linear Regression predicts **continuous** outcomes.
* Minimizes **MSE** via OLS.
* Solve via **Normal Equation** or **Gradient Descent**.
* **Regularize** (Ridge/Lasso) to combat overfitting.
* **Validate assumptions** with residual analysis.

---