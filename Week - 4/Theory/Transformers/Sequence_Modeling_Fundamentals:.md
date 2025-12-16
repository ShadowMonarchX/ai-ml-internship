## ✨ Sequence Modeling Fundamentals: RNN, LSTM, GRU, and Attention

This repository provides a comprehensive guide to the foundational concepts of **Sequence Modeling** in deep learning, covering Recurrent Neural Networks (RNNs), LSTMs, GRUs, and the critical Attention mechanism.

---

## 📘 Core Topics Covered

### 1. Recurrent Neural Networks (RNN)

RNNs are neural networks specifically designed to process sequential data. They operate by maintaining a **hidden state ($h_t$)** that acts as a form of memory, capturing information from all previous time steps.

**Key Concept:** The output at time $t$ is dependent on both the current input ($x_t$) and the hidden state from the previous time step ($h_{t-1}$).



[Image of Recurrent Neural Network architecture]


**Mathematical Formulation:**
$$h_t = f(W x_t + U h_{t-1} + b)$$

* $x_t$ : Input vector at time $t$
* $h_t$ : Hidden state (memory) vector at time $t$
* $W, U, b$ : Learnable weight matrices and bias vector
* $f$ : Non-linear activation function (e.g., $\tanh$)

**Limitations:** The repeated matrix multiplications lead to the **Vanishing Gradient Problem** for long sequences, making it difficult for the model to capture **long-term dependencies**.

---

### 2. Long Short-Term Memory (LSTM)

LSTMs were introduced to address the vanishing gradient problem in RNNs by using a more complex internal structure that includes a **Cell State ($C_t$)** and specialized regulatory mechanisms called **Gates**.

**Key Concept:** The Cell State ($C_t$) runs straight through the unit, facilitating the smooth flow of gradient information over long distances. The gates control the flow of information *into* and *out of* the cell state.



* **Forget Gate ($f_t$):** Decides which information to discard from the previous cell state.
    $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
* **Input Gate ($i_t$):** Decides what new information to store in the cell state.
    $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
* **Candidate Memory ($\tilde{C}_t$):** A potential new cell state vector.
    $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
* **Cell Update:** The mechanism for combining the old and new information.
    $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
* **Output Gate ($o_t$):** Controls the final hidden state output.
    $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
    $$h_t = o_t * \tanh(C_t)$$

---

### 3. Gated Recurrent Unit (GRU)

GRUs are a simplified variation of LSTMs. They combine the cell state and the hidden state, and use only two gates: the **Update Gate ($z_t$)** and the **Reset Gate ($r_t$)**.



**Key Concept:** GRUs achieve performance comparable to LSTMs on many tasks while having fewer parameters, leading to faster training and requiring less data.

* **Update Gate:** Determines how much of the past information (from $h_{t-1}$) needs to be passed to the future.
* **Reset Gate:** Determines how much of the past hidden state to forget.

---

### 4. Attention Mechanism

Attention is a technique that allows a model to weigh the importance of different parts of the input sequence when generating a specific part of the output sequence.

**Key Concept:** Instead of compressing the entire source sequence into a single fixed-length vector (like traditional Seq2Seq RNNs), Attention uses a weighted sum of all source hidden states, where the weights are dynamically calculated based on the current target state.

**Scaled Dot-Product Attention (Used in Transformers):**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

* $Q$ (Query), $K$ (Key), $V$ (Value) : Matrices representing the different roles of the input vectors.
* $QK^T$ : Measures the compatibility (score) between the Query and all Keys.
* $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ : Calculates the **Attention Weights** (the $\sqrt{d_k}$ term is for scaling).
* The final output is the weighted sum of the Value vectors.

**Advantages:**
* Solves the bottleneck problem of the fixed-length vector in traditional Seq2Seq.
* Enables the model to capture **true long-range dependencies** efficiently.

---

## 📌 References

* **Understanding LSTM Networks:** *A classic blog post providing great visualization and intuition.*
    * [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* **Original LSTM Paper:** *Long Short-Term Memory*
    * [S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory," Neural Computation 9(8):1735-1780, 1997.](https://www.bioinf.jku.at/publications/older/2604.pdf)
* **Original GRU Paper:** *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*
    * [K. Cho et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling," EMNLP, 2014.](https://arxiv.org/abs/1412.3555)
* **Original Attention Paper (Seq2Seq):** *Neural Machine Translation by Jointly Learning to Align and Translate*
    * [D. Bahdanau, K. Cho, Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," ICLR, 2015.](https://arxiv.org/abs/1409.0473)
* **Transformer Architecture (Self-Attention):** *Attention Is All You Need*
    * [A. Vaswani et al., "Attention Is All You Need," NIPS, 2017.](https://arxiv.org/abs/1706.03762)