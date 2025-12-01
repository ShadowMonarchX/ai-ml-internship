# 📘 Linear Regression — Full Theoretical + Mathematical Notes

---

## Overview
This repository contains **final study notes** for *Linear Regression* (deep theoretical explanation with maths) and a **ready-to-copy Python implementation** you can paste into your Git repo. The README is formatted for easy reading and direct use as study material.


## Table of Contents
1. [What is Linear Regression?](#what-is-linear-regression)
2. [Types of Linear Regression](#types-of-linear-regression)
3. [Assumptions](#assumptions-of-linear-regression)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Cost Function — MSE](#cost-function--mse)
6. [Finding Coefficients (Normal Equation & Gradient Descent)](#finding-best-coefficients)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Gradient Descent Variants](#gradient-descent-variants)
9. [Problems & Remedies](#problems-with-linear-regression)
10. [Regularization (Ridge, Lasso, ElasticNet)](#regularization)
11. [Geometric & Statistical Interpretation](#geometric-and-statistical-interpretation)
12. [Residual Analysis](#residual-analysis)
13. [When to Use / Not Use](#when-to-use-linear-regression)
14. [Quick Summary](#summary)
15. [Python Implementation (copy-paste)](#python-implementation)

---

## 1. What is Linear Regression?
Linear Regression is a **supervised learning algorithm** used to model the relationship between:

- **Independent variables (features)** → `X`
- **Dependent variable (target)** → `y`

**Goal:** Find a best-fit straight line (or hyperplane) that predicts `y` from `X`.

---

## 2. Types of Linear Regression

### 2.1 Simple Linear Regression
One feature:

\[
y = \beta_0 + \beta_1 x + \varepsilon
\]

### 2.2 Multiple Linear Regression
Multiple features:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon
\]

### 2.3 Polynomial Regression
Handle non-linear relations by adding polynomial features (still linear in coefficients).

---

## 3. Assumptions of Linear Regression (Very Important)
1. **Linearity** — relationship between features and target is linear.
2. **Independence** — observations are independent.
3. **Homoscedasticity** — constant variance of residuals.
4. **Normality of errors** — residuals approximately normally distributed.
5. **No multicollinearity** — features are not highly correlated.

If these fail, statistical inference becomes unreliable.

---

## 4. Mathematical Formulation
Matrix notation:

\[
y = X\beta + \varepsilon
\]

- `X` — feature matrix (m × n)
- `\beta` — coefficient vector
- `y` — target vector
- `\varepsilon` — error term (noise)

---

## 5. Cost Function — Mean Squared Error (MSE)

\[
J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
\]

Where `\hat{y} = X\beta`. The goal is to minimize `J(\beta)`.

---

## 6. Finding Best Coefficients

### 6.1 Normal Equation (Closed-form)

\[
\beta = (X^T X)^{-1} X^T y
\]

Pros: exact solution, no learning rate. Cons: expensive for large n, requires invertible `X^TX`.

### 6.2 Gradient Descent (Iterative)

Gradient of the cost:

\[
\nabla_{\beta} J = -\frac{1}{m} X^T (y - X\beta)
\]

Update rule (vectorized):

\[
\beta := \beta + \alpha \frac{1}{m} X^T (y - X\beta)
\]

`\alpha` is the learning rate. Repeat till convergence.

---

## 7. Evaluation Metrics
- **R² (Coefficient of Determination):**
  \[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \]
  where `SS_res = \sum (y - \hat{y})^2`, `SS_tot = \sum (y - \bar{y})^2`.

- **Adjusted R²:** penalizes additional features.

- **MSE / RMSE / MAE:** common error metrics.

---

## 8. Gradient Descent Variants
1. **Batch GD:** uses all training data per update.
2. **Stochastic GD:** uses one sample per update.
3. **Mini-batch GD:** uses small batches (common practice).

---

## 9. Problems with Linear Regression
- **Outliers** can heavily influence the fit.
- **Multicollinearity** → unstable coefficient estimates.
- **Underfitting** when relation is non-linear.

Remedies: robust regression, remove/transform outliers, regularization, feature engineering.

---

## 10. Regularization
Used to reduce overfitting and handle multicollinearity.

- **Ridge (L2):**
  \[ Loss = MSE + \lambda \sum_{j} \beta_j^2 \]

- **Lasso (L1):**
  \[ Loss = MSE + \lambda \sum_{j} |\beta_j| \]

- **ElasticNet:** mix of L1 and L2.

---

## 11. Geometric & Statistical Interpretation
- Geometric: find a hyperplane minimizing squared perpendicular distances.
- Statistical: under the assumption `\varepsilon \sim N(0, \sigma^2)`, OLS estimates = MLE.

Simple 1D slope formula:

\[
\beta_1 = \frac{Cov(X,Y)}{Var(X)}, \qquad \beta_0 = \bar{y} - \beta_1 \bar{x}
\]

---

## 12. Residual Analysis
Residuals: `e_i = y_i - \hat{y}_i`.

Good model: residuals are randomly scattered, constant variance, no visible pattern.

Use residual plots, Q–Q plots, and tests to validate assumptions.

---

## 13. When to Use / Not Use
**Use when:** linear relation, interpretability required, no severe outliers.

**Avoid when:** complex non-linear patterns, high multicollinearity, many categorical features without encoding.

---

## 14. Summary (Revision)
- Linear Regression predicts continuous outcomes.
- Minimizes MSE via OLS.
- Solve via Normal Equation or Gradient Descent.
- Regularize to combat overfitting.
- Validate assumptions with residual analysis.

---
