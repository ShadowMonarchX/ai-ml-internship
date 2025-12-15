# 📷 Convolutional Neural Networks (CNN) — Theoretical Foundation

---

## Overview

A **Convolutional Neural Network (CNN)** is a specialized type of Artificial Neural Network (ANN) designed primarily for processing data that has a known grid-like topology, such as **image data** (2D grid of pixels) or time-series data (1D sequence). CNNs leverage spatial locality by using shared weights, making them highly efficient for tasks like image recognition and computer vision.

---

## 🧭 Table of Contents

1. [Why CNNs for Images?](#1-why-cnns-for-images)
2. [CNN Architecture: The Layers](#2-cnn-architecture-the-layers)
3. [Convolutional Layer: The Feature Extractor (CONV)](#3-convolutional-layer-the-feature-extractor-conv)
4. [Pooling Layer: Downsampling (POOL)](#4-pooling-layer-downsampling-pool)
5. [Forwardpropagation: Math of Feature Maps](#5-forwardpropagation-math-of-feature-maps)
6. [Backpropagation: Training the Convolutional Layer (Advanced)](#6-backpropagation-training-the-convolutional-layer-advanced)
7. [Fully Connected Layer (FC)](#7-fully-connected-layer-fc)

---

## 1. Why CNNs for Images?

Standard Feedforward ANNs treat an image as a single, flattened vector of pixels. This leads to an **explosion of parameters** for high-resolution images. CNNs solve this by enforcing **local connectivity** and **weight sharing**.

* For a $200 \times 200 \times 3$ image (120,000 pixels), a fully connected layer with 1,000 neurons would require 120 million weights.
* CNNs use **local receptive fields** and **shared weights** (kernels) to keep the parameter count manageable and exploit the spatial structure of the data.

| Principle | Description | Advantage |
| :--- | :--- | :--- |
| **Local Connectivity** | Each neuron in $\text{CONV}$ layer connects only to a small, localized region of the input volume. | Fewer connections, local feature detection (e.g., edges). |
| **Parameter Sharing** | The same set of weights (**kernel**) is used across the entire spatial extent of the input. | **Dramatically reduced parameter count** and **translational invariance**. |
| **Equivariance** | If the input changes (e.g., shifts), the feature map output changes in the same way (shifts). | Helps in robust pattern recognition regardless of object position. |

---

## 2. CNN Architecture: The Layers

A typical CNN architecture consists of a sequence of feature extraction layers followed by a classifier. Unlike standard ANNs where neurons are fully connected, CNN layers are designed to exploit the **spatial structure** of data like images. 

### 2.1 Layer Types

1.  **Convolutional Layer ($\text{CONV}$):** The core layer; applies filters to the input to create feature maps.
2.  **Activation Layer ($\text{ReLU}$):** Applies a non-linear activation function element-wise (e.g., $f(x) = \max(0, x)$).
3.  **Pooling Layer ($\text{POOL}$):** Downsamples the feature maps, reducing spatial size and parameters.
4.  **Fully Connected Layer ($\text{FC}$):** Flattens the final feature maps and connects to a standard ANN classifier for final decision-making.

---

## 3. Convolutional Layer: The Feature Extractor ($\text{CONV}$)

The $\text{CONV}$ layer is where the network learns to detect features (edges, corners, textures). It uses a set of trainable filters (kernels) to scan the entire input.

### 3.1 The Convolution Operation

The convolution operation slides a small **kernel matrix** ($\mathbf{K}$, containing the layer's weights) over the input volume ($\mathbf{I}$), performing element-wise multiplication and summation to produce a single output value in the **Feature Map** ($\mathbf{S}$). 

$$S(i, j) = (\mathbf{I} * \mathbf{K})(i, j) + b = \sum_{u} \sum_{v} I(i-u, j-v) K(u, v) + b$$

* The same kernel weights are **shared** across the entire input, which dramatically reduces the number of parameters.
* The **bias** ($b$) is a single value added to the entire resulting feature map.

### 3.2 Hyperparameters and Output Size Calculation

The size of the output feature map ($N_{\text{out}}$) is critical for network design.

$$N_{\text{out}} = \left\lfloor \frac{N_{\text{in}} + 2P - F}{S} \right\rfloor + 1$$

| Hyperparameter | Symbol | Description |
| :--- | :--- | :--- |
| **Input Size** | $N_{\text{in}}$ | Spatial dimension of the input (Width or Height). |
| **Filter Size** | $F$ | Dimensions of the kernel (e.g., $3 \times 3$). |
| **Padding** | $P$ | Number of zero-value pixels added to borders. |
| **Stride** | $S$ | Number of pixels the filter shifts at each step. |
| **Depth (Filters)** | $K$ | Number of kernels (**determines output depth**). |

#### 💡 Example: $\text{CONV}$ Layer Output

| Parameters | Value |
| :--- | :--- |
| **Input** ($N_{\text{in}}$) | $6 \times 6$ image |
| **Filter** ($F$) | $3 \times 3$ |
| **Padding** ($P$) | 0 (No padding) |
| **Stride** ($S$) | 1 |

Using the formula:
$$N_{\text{out}} = \left\lfloor \frac{6 + 2(0) - 3}{1} \right\rfloor + 1 = 4$$

The output feature map will be **$4 \times 4$**.

---

## 4. Pooling Layer: Downsampling ($\text{POOL}$)

The $\text{POOL}$ layer is used to spatially reduce the size of the feature maps, making the network more robust to small shifts in the input data (**translational invariance**) and reducing computation. It has **no learnable parameters** (weights or biases). 

### 4.1 Pooling Operation

Pooling operates independently on every depth slice (feature map) of the input.

#### a. Max Pooling (Most Common)
Selects the **maximum value** within the pooling window, preserving the most salient feature response.

$$A_{i, j, d}^{\text{pooled}} = \max_{u=1}^{F} \max_{v=1}^{F} A_{\text{local}}^{\text{prev}}$$

#### b. Average Pooling
Selects the **average value** within the pooling window.

### 4.2 Output Size Calculation

The output size calculation uses the same formula as the $\text{CONV}$ layer, typically with $P=0$:

$$N_{\text{out}} = \left\lfloor \frac{N_{\text{in}} - F}{S} \right\rfloor + 1$$

#### 💡 Example: Max Pooling Output

| Parameters | Value |
| :--- | :--- |
| **Input** ($N_{\text{in}}$) | $4 \times 4$ feature map |
| **Filter** ($F$) | $2 \times 2$ |
| **Padding** ($P$) | 0 |
| **Stride** ($S$) | 2 |

Using the formula:
$$N_{\text{out}} = \left\lfloor \frac{4 - 2}{2} \right\rfloor + 1 = 2$$

The output pooled map will be **$2 \times 2$**. The input volume of $4 \times 4$ has been reduced by a factor of 4.

---

## 5. Forwardpropagation: Math of Feature Maps

The process of Forwardpropagation involves sequential application of the convolution and activation functions, transforming the input volume through the layers.

### Step 1: Convolution and Bias
$$\mathbf{Z}^{[l]} = \text{Conv}(\mathbf{A}^{[l-1]}, \mathbf{W}^{[l]}) + \mathbf{b}^{[l]}$$

### Step 2: Activation
$$\mathbf{A}^{[l]} = g(\mathbf{Z}^{[l]})$$

### Step 3: Pooling (If present)
$$\mathbf{A}_{\text{pooled}}^{[l]} = \text{Pool}(\mathbf{A}^{[l]})$$

The dimensions are governed by the output size formula in Section 3.2.

---

## 6. Backpropagation: Training the Convolutional Layer (Advanced)

Training CNNs relies on **Gradient Descent** guided by **Backpropagation** using the **Chain Rule**. The process must specifically account for **weight sharing** and the **non-linear sampling** in pooling.

### 6.1 Backpropagating Through the $\text{POOL}$ Layer ($\frac{\partial \mathcal{L}}{\partial \mathbf{A}}$)

The error ($\delta$) is passed backward from the pooled output ($\mathbf{A}_{\text{pooled}}^{[l]}$) to the pre-pooled input ($\mathbf{A}^{[l]}$) via upsampling.

* **Max Pooling:** The gradient is passed **only to the single neuron position** that held the maximum value in the forward pass. All other positions receive a zero gradient. This requires storing a "mask" (switch variables) during the forward pass.
* **Average Pooling:** The gradient is **distributed equally** among all neurons in the pooling window ($\frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[l]}} \leftarrow \frac{\partial \mathcal{L}}{\partial \mathbf{A}_{\text{pooled}}^{[l]}} / F^2$).

### 6.2 Backpropagating Through the $\text{CONV}$ Layer

The core step is calculating the gradients for the shared weights ($\mathbf{W}$) and the previous input ($\mathbf{A}^{[l-1]}$).

#### a. Gradient with Respect to Weights ($\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$)

To update the shared weights $\mathbf{W}$ of a kernel, the local gradients are **summed** across all spatial locations where the kernel was applied:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \sum_{\text{locations}} \left( \text{Input Patch} \odot \delta^{[l]} \right)$$

* This summation operation is mathematically equivalent to a **valid convolution** between the previous layer's input ($\mathbf{A}^{[l-1]}$) and the error sensitivity ($\delta^{[l]}$).

#### b. Gradient with Respect to Input ($\frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[l-1]}}$)

To propagate the error further backward to the previous layer, the error sensitivity ($\delta^{[l]}$) is convolved with the **flipped** kernel weights:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[l-1]}} = \text{Full Convolution}\left( \text{Pad}(\delta^{[l]}), \text{Rotated } \mathbf{W}^{[l]} \right)$$

* The operation is mathematically known as **cross-correlation** (or **full convolution**).

### 6.3 Parameter Update (Gradient Descent)

The shared kernel weights $\mathbf{W}$ and biases $\mathbf{b}$ are updated using the calculated gradients:

$$\mathbf{W}^{[l]} = \mathbf{W}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$$

$$\mathbf{b}^{[l]} = \mathbf{b}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$$

---

## 7. Fully Connected Layer (FC)

The final stage of the network transforms the extracted features into a classification or regression result.

### 7.1 Flattening
The output of the final $\text{CONV}$ or $\text{POOL}$ layer (e.g., $4 \times 4 \times 128$ volume) is converted into a $1$D feature vector.

### 7.2 FC Layers
This flattened vector is passed to one or more standard **Fully Connected (FC)** layers. These layers act as a standard ANN classifier.

* **Input:** The flattened vector containing the high-level, spatially invariant features learned by the CNN layers.
* **Output:** The final $\text{FC}$ layer usually uses a **Softmax** activation for multi-class tasks to output the probability distribution over all classes.

### 7.3 Backpropagation in FC Layers
Standard ANN backpropagation rules (treating each feature map element as an individual neuron output) apply directly to these layers, as they contain no weight sharing.