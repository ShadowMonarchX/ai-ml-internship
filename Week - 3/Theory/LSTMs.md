# 🔄 Recurrent Neural Networks (RNN) and LSTMs — Theory and Implementation

---

## Overview

**Recurrent Neural Networks (RNNs)** are a class of Artificial Neural Networks designed specifically to handle **sequential data** (such as text, speech, or time series). Unlike traditional Feedforward Networks (ANNs/CNNs), RNNs possess an internal memory that allows them to use information from previous steps in the sequence to influence the current output.

---

## 🧭 Table of Contents

1. [The Challenge of Sequence Data](#1-the-challenge-of-sequence-data)
2. [Recurrent Neural Networks (RNNs)](#2-recurrent-neural-networks-rnns)
3. [The Problem: Vanishing/Exploding Gradients](#3-the-problem-vanishingexploding-gradients)
4. [Long Short-Term Memory (LSTM)](#4-long-short-term-memory-lstm)
5. [Forwardpropagation in LSTMs](#5-forwardpropagation-in-lstms)
6. [Gated Recurrent Unit (GRU)](#6-gated-recurrent-unit-gru)

---

## 1. The Challenge of Sequence Data

Traditional ANNs/CNNs treat all inputs as independent. This fails for sequences where the order and context matter (e.g., in a sentence, the meaning of a word depends on preceding words).

RNNs overcome this by introducing a **recurrent connection** that feeds the output of the hidden layer from the previous time step ($t-1$) back into the current time step ($t$).

## 2. Recurrent Neural Networks (RNNs)

An RNN processes inputs one step at a time, maintaining a **Hidden State** ($h_t$) that acts as the network's memory.

### 2.1 The Math of the Hidden State

At each time step $t$, the new hidden state $h_t$ is calculated based on the current input $x_t$ and the previous hidden state $h_{t-1}$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

The output $y_t$ is then calculated from the new hidden state $h_t$:

$$y_t = W_{hy} h_t + b_y$$

* $W_{hh}$: Weight matrix for the recurrent connection (hidden-to-hidden).
* $W_{xh}$: Weight matrix for the input-to-hidden connection.
* **Key Concept: Parameter Sharing:** The same weight matrices ($W_{hh}, W_{xh}, W_{hy}$) and biases ($b_h, b_y$) are used across **all time steps**. This is fundamental to RNNs.



### 2.2 Training: Backpropagation Through Time (BPTT)

RNNs are trained using **Backpropagation Through Time (BPTT)**. This is a modification of standard backpropagation where the loss gradient is calculated and propagated backward through the network's unrolled structure across the entire sequence.

## 3. The Problem: Vanishing/Exploding Gradients

While training with BPTT, the multiplication of gradients over many time steps (long sequences) leads to two critical issues:

* **Vanishing Gradients:** Gradients become infinitesimally small, making the network unable to learn long-term dependencies (i.e., information from early time steps is "forgotten" quickly).
* **Exploding Gradients:** Gradients become excessively large, leading to unstable training and large weight updates. (This can be partially addressed with **Gradient Clipping**).

This inability to capture long-range context led to the development of **LSTMs**.

---

## 4. Long Short-Term Memory (LSTM)

**LSTMs** are a special kind of RNN designed to explicitly address the vanishing gradient problem, enabling them to learn **long-term dependencies**. LSTMs use a more complex structure called a **Cell** that contains three explicit **Gates** to regulate the flow of information.



### 4.1 The Three Gates

Each gate is a layer using the **Sigmoid** activation function ($\sigma$). Since the Sigmoid output is between 0 and 1, the gate acts as a valve: a value close to 0 means "let nothing pass," and a value close to 1 means "let everything pass."

1.  **Forget Gate ($f_t$):** Decides what information from the previous **Cell State** ($C_{t-1}$) should be discarded.
    $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2.  **Input Gate ($i_t$):** Decides which new information will be stored in the cell state. This is done in two parts:
    * $i_t$: The sigmoid layer decides which values to update.
    * $\tilde{C}_t$: The candidate vector (using $\tanh$) creates potential new values.
    $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
    $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

3.  **Output Gate ($o_t$):** Decides what value from the Cell State will be outputted as the new **Hidden State** ($h_t$).
    $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

---

## 5. Forwardpropagation in LSTMs

The magic of LSTMs lies in the **Cell State** ($C_t$), which flows straight through the cell with only minor linear interactions, making it easier for gradients to flow backward without vanishing.

### 5.1 Updating the Cell State ($C_t$)

The new Cell State is calculated by two operations: forgetting the past and adding the new candidate information.

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

* $f_t \odot C_{t-1}$: Old state $C_{t-1}$ multiplied by the Forget Gate $f_t$ (what to keep).
* $i_t \odot \tilde{C}_t$: Candidate information $\tilde{C}_t$ multiplied by the Input Gate $i_t$ (what to add).
* $\odot$: Element-wise multiplication.

### 5.2 Updating the Hidden State ($h_t$)

The new Hidden State (which serves as the output and memory for the next cell) is based on the filtered Cell State.

$$h_t = o_t \odot \tanh(C_t)$$

* The Cell State $C_t$ is passed through $\tanh$ (squashing the values between -1 and 1) and then multiplied by the Output Gate $o_t$ (determining what filtered part of the memory is exposed).

---

## 6. Gated Recurrent Unit (GRU)

The **Gated Recurrent Unit (GRU)** is a slightly simplified and more computationally efficient variation of the LSTM. It achieves similar performance but uses only two gates, combining the Forget and Input gates into an **Update Gate**.

### 6.1 Key Differences from LSTM

* **Two Gates:** It uses an **Update Gate** ($z_t$) and a **Reset Gate** ($r_t$).
* **No Explicit Cell State:** The Hidden State $h_t$ and the Cell State $C_t$ are merged.
* **Fewer Parameters:** GRUs converge faster on smaller datasets due to fewer parameters.

### 6.2 The GRU Gates

1.  **Update Gate ($z_t$):** Controls how much of the previous hidden state ($h_{t-1}$) should be carried forward.
    $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

2.  **Reset Gate ($r_t$):** Controls how much of the previous hidden state should be forgotten.
    $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

### 6.3 GRU Update

The new hidden state $h_t$ is a linear combination of the previous state and the new candidate state $\tilde{h}_t$.

$$\tilde{h}_t = \tanh(W_{\tilde{h}} \cdot [r_t \odot h_{t-1}, x_t])$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$