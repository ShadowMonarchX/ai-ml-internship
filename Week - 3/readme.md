
---

# 🧠 PyTorch ANN Practical Task – Full Project

### **Forward + Backward Propagation | Student Performance Prediction**

---

## **🎯 Objective**

Build a simple **Artificial Neural Network (ANN)** using **PyTorch** to predict **FinalGrade** from selected student features.
You will clearly understand:

* Forward Propagation
* Loss Calculation
* Backward Propagation (manual gradient intuition)
* Weight Updates
* ANN training workflow

---

# 1️⃣ Dataset Selection

We will use your uploaded dataset **merged_dataset.csv** which includes:

* StudyHours
* Attendance
* Motivation
* Extracurricular
* Learning Style
* Stress Level
* ExamScore
* FinalGrade (Target)

### **Your Task**

✔ Select **3–5 input features** you want to use for prediction.
✔ Target variable = **FinalGrade**.

> Example:
> Input features → `StudyHours, Attendance, Motivation, ExamScore`

---

# 2️⃣ Network Architecture Design

Design a **simple feed-forward neural network**:

### **Architecture**

* **Input Layer:** Number of neurons = number of selected features
* **Hidden Layer:** 3–5 neurons
* **Activation:** ReLU
* **Output Layer:** 1 neuron
* **Activation:** Linear (Regression)

### **Your Task**

* Decide **how many hidden neurons** to use
* Draw a simple diagram:

```
Input features → Hidden Layer (ReLU) → Output Layer (FinalGrade)
```

---

# 3️⃣ Forward Propagation

For each layer:

## **Step 1 — Weighted Sum**

For every neuron:

[
Z = W \cdot X + b
]

* (W) = weights
* (X) = input features
* (b) = bias

---

## **Step 2 — Activation Function**

Hidden layer uses **ReLU**:

[
A = \text{ReLU}(Z) = \max(0, Z)
]

Output layer uses **linear activation**:

[
\hat{y} = Z_{output}
]

---

## **Your Task**

Take **one row from the dataset**, manually compute:

1. (Z) for each hidden neuron
2. Apply ReLU
3. Compute output neuron value
4. Record predicted value (\hat{y})

---

# 4️⃣ Loss Calculation

We calculate **regression error** using **Mean Squared Error (MSE)**:

[
L = \frac{1}{n} \sum (y - \hat{y})^{2}
]

Where:

* (y) = actual FinalGrade
* (\hat{y}) = predicted FinalGrade

### **Your Task**

Compute MSE for **one training example** manually.

---

# 5️⃣ Backward Propagation

(How ANN learns by correcting errors)

---

## **Step 1 — Gradient at Output Layer**

[
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})
]

---

## **Step 2 — Gradient for Output Layer Weights**

[
\frac{\partial L}{\partial W_{output}} = A_{hidden} \cdot \frac{\partial L}{\partial \hat{y}}
]

Where:

* (A_{hidden}) = outputs of hidden layer neurons

---

## **Step 3 — Backprop to Hidden Layer**

Hidden layer error:

[
\delta_{hidden} = (W_{output})^T \cdot \delta_{output} ;; \odot ;; f'(Z_{hidden})
]

Since ReLU:

[
f'(Z) =
\begin{cases}
1 & \text{if } Z > 0 \
0 & \text{otherwise}
\end{cases}
]

---

## **Step 4 — Gradients for Hidden Weights**

[
\frac{\partial L}{\partial W_{hidden}} = X \cdot \delta_{hidden}
]

---

# 6️⃣ Updating Weights (Gradient Descent)

[
W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}
]

[
b_{new} = b_{old} - \eta \cdot \frac{\partial L}{\partial b}
]

Where:

* (\eta) = learning rate (e.g., 0.01)

### **Your Task**

Using the gradients from Step 5:

* Update hidden layer weights
* Update output layer weights
* Update biases

---

# 7️⃣ Training Loop (Iteration)

During training:

1. Forward pass
2. Compute loss
3. Backward pass (gradients)
4. Update all weights and biases
5. Repeat for **multiple epochs**

### **Your Task**

* Train for **20–50 epochs**
* Record loss every 5 epochs
* Write observations:

  * Did the loss decrease?
  * Did predictions improve?

---

# 8️⃣ Result & Analysis

After training:

### **Your Task**

#### ✔ Final Predictions

Compare:

| Actual Value (FinalGrade) | Predicted Value |
| ------------------------- | --------------- |

#### ✔ Loss Curve (optional)

Plot / Describe:

```
Epoch vs Loss
```

#### ✔ Final Conclusion

Explain in your own words:

* How forward propagation produced predictions
* How backward propagation corrected weights
* How the ANN learned from data

---

# 9️⃣ Final Deliverables (Submit)

You must submit:

### 🔹 1. ANN Architecture Diagram

(input → hidden → output)

### 🔹 2. Forward Propagation (manual example)

### 🔹 3. Backpropagation (manual gradient steps)

### 🔹 4. ANN Training Description

(PyTorch training loop — no code required)

### 🔹 5. Loss Trend

### 🔹 6. Final Predictions vs Actual

---
