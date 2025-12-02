# 📘 Linear Regression — Full Theoretical + Mathematical Notes

## 1️⃣ What is Linear Regression?

<<<<<<< HEAD
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

=======
>>>>>>> 3646266 (Update linear_regression_theory.md)
Linear Regression is a **supervised learning algorithm** used to model the relationship between:

* **Independent variables (features)** → $X$
* **Dependent variable (target)** → $y$
<<<<<<< HEAD

**Goal:** Find a **best-fit straight line (or hyperplane)** that predicts $y$ from $X$.


[Image of Simple Linear Regression best-fit line plotted on scatter data]

=======

**Goal:**
👉 Find a **best-fit straight line** that predicts $y$ from $X$.
>>>>>>> 3646266 (Update linear_regression_theory.md)

---

## 2️⃣ Types of Linear Regression

<<<<<<< HEAD
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
=======
### **1. Simple Linear Regression**
* One feature
* **Model:**
$$y = \beta_0 + \beta_1 x + \varepsilon$$

### **2. Multiple Linear Regression**
* Multiple features
* **Model:**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon$$

### **3. Polynomial Regression**
* Non-linear relation handled with polynomial features
* Still linear in coefficients.

---

## 3️⃣ Assumptions of Linear Regression (Very Important)

To get reliable results, Linear Regression assumes:

1. **Linearity:** Relationship between features and output is linear.
2. **Independence:** Observations are independent.
3. **Homoscedasticity:** Equal variance of errors.
4. **Normality of Errors:** Residuals ~ Normal distribution.
5. **No Multicollinearity:** Features should not be highly correlated.

---

## 4️⃣ Mathematical Formulation

### **Model Equation (Vector Form)**
For multiple regression:

$$y = X\beta + \varepsilon$$

**Where:**
* $X$ → matrix of features
* $\beta$ → coefficients
* $y$ → target
* $\varepsilon$ → error term
>>>>>>> 3646266 (Update linear_regression_theory.md)

---

## 5️⃣ Cost Function – Mean Squared Error (MSE)

<<<<<<< HEAD
$$\text{J}(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

Where $\hat{y} = X\beta$. The goal is to **minimize** $J(\beta)$.
=======
Linear Regression minimizes the **sum of squared errors**.

$$J(\beta) = \frac{1}{2m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2$$

**Where:**
* $m$ = number of samples
* $y_i$ = actual value
* $\hat{y}_i = X\beta$ = predicted value

**Goal:**
👉 **Minimize** $J(\beta)$
>>>>>>> 3646266 (Update linear_regression_theory.md)

---

## 6️⃣ Finding Best Coefficients ($\beta$)

<<<<<<< HEAD
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

* Linear Regression predicts **continuous** outcomes.
* Minimizes **MSE** via OLS.
* Solve via **Normal Equation** or **Gradient Descent**.
* **Regularize** (Ridge/Lasso) to combat overfitting.
* **Validate assumptions** with residual analysis.

---
=======
### **Method 1: Normal Equation**
Closed-form solution (no gradient descent needed):

$$\beta = (X^TX)^{-1}X^Ty$$

* **Works well when:** small dataset, features < 10,000
* **Fails when:** matrix becomes non-invertible, large dataset (slow)

### **Method 2: Gradient Descent**
Iterative optimization:

$$\beta := \beta - \alpha \frac{\partial J(\beta)}{\partial \beta}$$

**Where:**
* $\alpha$ = learning rate

**Compute gradient:**
$$\frac{\partial J}{\partial\beta} = -\frac{1}{m}X^T(y-X\beta)$$

**Update rule:**
$$\beta := \beta + \alpha \frac{1}{m}X^T(y-X\beta)$$

Repeat until convergence.

---

## 7️⃣ Evaluation Metrics

### **1. R² Score**
Measures how much variance in $y$ is explained.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

**Where:**
* $SS_{res} = \sum (y - \hat{y})^2$
* $SS_{tot} = \sum (y - \bar{y})^2$

### **2. Adjusted R²**
Penalizes extra features.

$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$$

**Where:**
* $n$ → samples
* $k$ → features

### **3. RMSE: Root Mean Squared Error**

$$RMSE = \sqrt{\frac{1}{m}\sum (y-\hat{y})^2}$$

---

## 8️⃣ Gradient Descent Variants

1. **Batch GD** – uses whole data
2. **Stochastic GD** – uses one example
3. **Mini-Batch GD** – uses small batches (most used)

---

## 9️⃣ Problems with Linear Regression

1. **Outliers influence model heavily**
2. **Multicollinearity → unstable coefficients**
3. **Underfitting if relationship is non-linear**

---

## 🔟 Regularization in Linear Regression

Used to reduce overfitting by penalizing large coefficients.

### **1. Ridge Regression (L2)**
$$J(\beta) = MSE + \lambda \sum\beta_i^2$$

### **2. Lasso Regression (L1)**
$$J(\beta) = MSE + \lambda \sum|\beta_i|$$

### **3. Elastic Net**
Combination of L1 + L2

---

## 1️⃣1️⃣ Geometric Interpretation

Linear Regression finds a **hyperplane** in n-dimensional space.

**Example:**
* 1 feature → line
* 2 features → plane
* n features → n-dimensional hyperplane

**Goal:** minimize perpendicular distance between points and that hyperplane.

---

## 1️⃣2️⃣ Statistical Interpretation

$$\beta_1 = \frac{Cov(X, Y)}{Var(X)}$$

**Intercept:**
$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

**This shows:**
* slope depends on covariance
* intercept shifts line to match mean

---

## 1️⃣3️⃣ Error / Residual Analysis

**Residual:**
$$e_i = y_i - \hat{y}_i$$

**Good model:**
* residuals randomly distributed
* no pattern
* constant variance

---

## 1️⃣4️⃣ When to Use Linear Regression

**Use when:**
✓ Relationship approx linear
✓ Data clean, no extreme outliers
✓ Interpretability needed

**Don't use when:**
✗ Complex non-linear relations
✗ High multicollinearity
✗ Many categorical variables without encoding

---

## 1️⃣5️⃣ Summary for Notes

* Linear Regression predicts output using straight line.
* Uses MSE cost function.
* Coefficients: Normal Equation / Gradient Descent.
* Evaluation: R², RMSE.
* Assumptions must be satisfied.
* Regularization prevents overfitting.
* Easy to interpret, fast, widely used.
>>>>>>> 3646266 (Update linear_regression_theory.md)
