
---

# **📘 Linear Regression — Full Theoretical + Mathematical Notes**


# **1️⃣ What is Linear Regression?**

Linear Regression is a **supervised learning algorithm** used to model the relationship between:

* **Independent variables (features)** → ( X )
* **Dependent variable (target)** → ( y )

Goal:
👉 Find a **best-fit straight line** that predicts ( y ) from ( X ).

---

# **2️⃣ Types of Linear Regression**

### **1. Simple Linear Regression**

* One feature
* Model:
          y=β0​+β1​x+ε

### **2. Multiple Linear Regression**

* Multiple features
* Model:
          y=β0​+β1​x1​+β2​x2​+⋯+βn​xn​+ε
### **3. Polynomial Regression**

* Non-linear relation handled with polynomial features
* Still linear in coefficients.

---

# **3️⃣ Assumptions of Linear Regression (Very Important)**

To get reliable results, Linear Regression assumes:

1. **Linearity**
   Relationship between features and output is linear.

2. **Independence**
   Observations are independent.

3. **Homoscedasticity**
   Equal variance of errors.

4. **Normality of Errors**
   Residuals ~ Normal distribution.

5. **No Multicollinearity**
   Features should not be highly correlated.

---

# **4️⃣ Mathematical Formulation**

### **Model Equation (Vector Form)**

For multiple regression:

          y=Xβ+ε

Where:

* ( X ) → matrix of features
* ( \beta ) → coefficients
* ( y ) → target
* ( \varepsilon ) → error term

---

# **5️⃣ Cost Function – Mean Squared Error (MSE)**

Linear Regression minimizes the **sum of squared errors**.

          J(β)=2m1​i=1∑m​(yi​−y^​i​)2

Where:

* 𝑚 = number of samples
* y_i  = actual value
* ( \hat{y}_i = X\beta ) = predicted value

Goal:
👉 **Minimize** ( J(\beta) )

---

# **6️⃣ Finding Best Coefficients (β)**

### **Method 1: Normal Equation**

Closed-form solution (no gradient descent needed):

          β=(XTX)−1XTy

Works well when:

* small dataset
* features < 10,000

Fails when:

* matrix becomes non-invertible
* large dataset → slow

---

### **Method 2: Gradient Descent**

Iterative optimization:

          β:=β−α∂β∂J(β)​

Where:

* ( \alpha ) = learning rate
* Compute gradient:

          [
          \frac{\partial J}{\partial\beta}=-\frac{1}{m}X^T(y-X\beta)
          ]

Update rule:

[
\beta := \beta + \alpha \frac{1}{m}X^T(y-X\beta)
]

Repeat until convergence.

---

# **7️⃣ Evaluation Metrics**

### **1. R² Score**

Measures how much variance in y is explained.

[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
]

Where:

* ( SS_{res} = \sum (y - \hat{y})^2 )
* ( SS_{tot} = \sum (y - \bar{y})^2 )

---

### **2. Adjusted R²**

Penalizes extra features.

[
R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}
]

Where:

* ( n ) → samples
* ( k ) → features

---

### **3. RMSE: Root Mean Squared Error**

[
RMSE = \sqrt{\frac{1}{m}\sum (y-\hat{y})^2}
]

---

# **8️⃣ Gradient Descent Variants**

1. **Batch GD** – uses whole data
2. **Stochastic GD** – uses one example
3. **Mini-Batch GD** – uses small batches (most used)

---

# **9️⃣ Problems with Linear Regression**

1. **Outliers influence model heavily**
2. **Multicollinearity → unstable coefficients**
3. **Underfitting if relationship is non-linear**

---

# **🔟 Regularization in Linear Regression**

Used to reduce overfitting by penalizing large coefficients.

### **1. Ridge Regression (L2)**

          J(β)=MSE+λ∑βi2​

### **2. Lasso Regression (L1)**

          J(β)=MSE+λ∑∣βi​∣

### **3. Elastic Net**

Combination of L1 + L2

---

# **1️⃣1️⃣ Geometric Interpretation**

Linear Regression finds a **hyperplane** in n-dimensional space.

Example:

* 1 feature → line
* 2 features → plane
* n features → n-dimensional hyperplane

Goal: minimize perpendicular distance between points and that hyperplane.

---

# **1️⃣2️⃣ Statistical Interpretation**

[
\beta_1 = \frac{Cov(X, Y)}{Var(X)}
]

Intercept:
[
\beta_0 = \bar{y} - \beta_1\bar{x}
]

This shows:

* slope depends on covariance
* intercept shifts line to match mean

---

# **1️⃣3️⃣ Error / Residual Analysis**

Residual =

          ei​=yi​−y^​i​

Good model:

* residuals randomly distributed
* no pattern
* constant variance

---

# **1️⃣4️⃣ When to Use Linear Regression**

Use when:
✓ Relationship approx linear
✓ Data clean, no extreme outliers
✓ Interpretability needed

Don't use when:
✗ Complex non-linear relations
✗ High multicollinearity
✗ Many categorical variables without encoding

---

# **1️⃣5️⃣ Summary for Notes**

* Linear Regression predicts output using straight line.
* Uses MSE cost function.
* Coefficients: Normal Equation / Gradient Descent
* Evaluation: R², RMSE
* Assumptions must be satisfied
* Regularization prevents overfitting
* Easy to interpret, fast, widely used

---


