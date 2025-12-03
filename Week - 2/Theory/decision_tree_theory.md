# 🌳 Decision Tree — Full Theoretical + Mathematical Notes

---

## Overview

This document provides a comprehensive theoretical and mathematical explanation of the **Decision Tree** algorithm, a fundamental non-parametric supervised learning method used for both classification and regression tasks.

---

## 🧭 Table of Contents

* [1. What is a Decision Tree?](#1-what-is-a-decision-tree)
* [2. Tree Components](#2-tree-components)
* [3. The Splitting Mechanism](#3-the-splitting-mechanism)
* [4. Measure of Impurity: Gini Index](#4-measure-of-impurity-gini-index)
* [5. Measure of Impurity: Entropy and Information Gain](#5-measure-of-impurity-entropy-and-information-gain)
* [6. Decision Tree Algorithms](#6-decision-tree-algorithms)
* [7. Overfitting and Pruning](#7-overfitting-and-pruning)
* [8. Advantages and Disadvantages](#8-advantages-and-disadvantages)

---

## 1. What is a Decision Tree?

A Decision Tree is a flowchart-like structure where each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label (for classification) or a numerical value (for regression).

* **Goal:** To create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
* **Method:** It partitions the feature space into a set of rectangles, and fits a simple constant model (the average class/value) in each.



[Image of Decision Tree structure]


---

## 2. Tree Components

1.  **Root Node:** The top-most node; represents the entire dataset, which subsequently gets split.
2.  **Internal Node:** Represents a feature test and holds the condition used to divide the data.
3.  **Branch/Edge:** Represents the outcome or decision resulting from the test at the node.
4.  **Leaf Node (Terminal Node):** Represents the final decision or prediction (the classification label or regression value).

---

## 3. The Splitting Mechanism

The most crucial step in building a Decision Tree is deciding which feature and which threshold should be used to split a node. This decision is based on maximizing the **Information Gain** or minimizing the **Impurity** of the resulting child nodes.

The algorithm chooses the split that results in the highest homogeneity (purity) in the resulting subsets.

### Homogeneity (Purity)

A node is considered pure (Impurity = 0) if all data points in that node belong to the same class.

---

## 4. Measure of Impurity: Gini Index

The **Gini Impurity** (used by the **CART** algorithm) measures the probability of misclassifying a randomly chosen element in the dataset if it were randomly labeled according to the distribution of labels in the node.

### 4.1 Gini Impurity Formula

For a node $N$ where $p_i$ is the proportion of samples belonging to class $i$:

$$I_G(N) = 1 - \sum_{i=1}^{C} (p_i)^2$$

* $C$: number of classes.
* The Gini Impurity is minimized when $I_G(N) \to 0$.

### 4.2 Gini Gain

When evaluating a split, we calculate the weighted average Gini Impurity of the resulting child nodes and subtract it from the parent's impurity. The split with the **maximum Gini Gain** is selected.

$$\text{Gini Gain} = I_G(\text{Parent}) - \sum_{k=1}^{K} \frac{N_k}{N_{\text{Parent}}} I_G(\text{Child}_k)$$

---

## 5. Measure of Impurity: Entropy and Information Gain

**Entropy** (used by the **ID3** and **C4.5** algorithms) quantifies the amount of randomness or uncertainty in a set of data. A perfectly mixed sample has high entropy, and a pure sample has zero entropy.

### 5.1 Entropy Formula

For a node $N$ with $C$ classes:
$$H(N) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

* $p_i$: proportion of samples belonging to class $i$ at the node.
* Entropy ranges from 0 (pure) to 1 (maximum disorder/uncertainty for binary classes).



### 5.2 Information Gain (IG)

**Information Gain** is the reduction in entropy achieved by splitting the data based on a particular feature. The goal is to maximize IG.

$$\text{IG}(\text{Split}) = H(\text{Parent}) - \sum_{k=1}^{K} \frac{N_k}{N_{\text{Parent}}} H(\text{Child}_k)$$

---

## 6. Decision Tree Algorithms

Different algorithms use different metrics and handling for data types:

| Algorithm | Criterion | Data Types | Notes |
| :--- | :--- | :--- | :--- |
| **ID3** | Entropy/IG | Categorical | Cannot handle numerical data. |
| **C4.5** | Gain Ratio | Categorical & Numerical | Improvement over ID3, handles missing values. |
| **CART** | Gini Index (Classification) / MSE (Regression) | Categorical & Numerical | The most common algorithm; produces binary splits (two children per node). |

### Regression Trees

For regression, the impurity measure used to select splits is typically the **Variance Reduction** or **Mean Squared Error (MSE)**. The goal is to minimize the sum of squared residuals in the child nodes.

$$\text{MSE}(N) = \frac{1}{|N|} \sum_{i \in N} (y_i - \bar{y})^2$$

---

## 7. Overfitting and Pruning

Decision Trees, especially deep ones, are highly prone to overfitting the training data, capturing noise and specific details that don't generalize well.

### 7.1 Stopping Conditions (Pre-Pruning)

Tree growth is halted prematurely based on criteria:

* Maximum tree depth is reached.
* Number of samples in a node falls below a threshold (e.g., `min_samples_split`).
* Impurity or Information Gain falls below a threshold.

### 7.2 Post-Pruning

The tree is grown to its maximum depth first, and then unnecessary branches and nodes are removed or collapsed using validation data to improve generalization. (e.g., Cost-Complexity Pruning).

---

## 8. Advantages and Disadvantages

| Advantages | Disadvantages |
| :--- | :--- |
| **Interpretability:** Easy to explain to non-technical users (white box model). | **Overfitting:** Prone to learning noise in the data, leading to poor generalization. |
| **Non-Linearity:** Requires little data preparation and handles non-linear relationships well. | **Instability (High Variance):** Small variations in the data can result in a completely different tree structure. |
| **Data Types:** Can handle both numerical and categorical data naturally. | **Bias:** Greedy search algorithm often fails to find the globally optimal tree. |