# Transformer Architecture: A Deep Dive into the Components

## PART 1: Input Representation & Encoding

This stage transforms raw text into a format the model can process mathematically.

### I. Tokenization

The process of breaking down raw text into smaller units (tokens). Modern Transformers typically use sub-word tokenization (e.g., Byte Pair Encoding or WordPiece) to handle rare words and balance vocabulary size.

### II. Token Embedding

* **Learned Embeddings:** Maps each token ID to a high-dimensional vector of size d_{model}. These vectors represent the semantic meaning of the words.
* **Shared / Tied Embeddings:** A technique where the weights of the input embedding layer and the final output projection layer are shared to reduce parameters.

### III. Positional Information

Since Transformers process tokens in parallel, they lack an inherent sense of order.

* **Sinusoidal Positional Encoding:** Uses fixed sine and cosine functions of different frequencies.
* **Learned Positional Embedding:** Treats positions as weights to be learned during training.
* **Rotary Positional Embedding (RoPE):** Encodes absolute position with a rotation matrix and naturally incorporates relative position.
* **ALiBi (Attention with Linear Biases):** Biases the attention scores based on the distance between tokens, allowing for better context length extrapolation.

### IV. Input Composition

The final input to the first layer is the element-wise sum of the **Token Embedding** and the **Positional Information**:


---

## PART 2: Attention Mechanism (Context Formation)

The "heart" of the Transformer that allows tokens to interact with one another.

### I. Query–Key–Value Projection

For each token, the model creates three vectors by multiplying the input by learned weight matrices W^Q, W^K, W^V:

* **Query (Q):** What I am looking for.
* **Key (K):** What I contain.
* **Value (V):** What information I provide once matched.

### II. Scaled Dot-Product Attention

Calculates the alignment between Queries and Keys, scaled to prevent gradient vanishing:


### III. Attention Types

* **Self-attention:** Tokens in a sequence attend to all other tokens in the same sequence.
* **Masked self-attention:** Used in decoders to prevent tokens from "looking ahead" at future tokens.
* **Cross-attention:** Queries come from one sequence (decoder), while Keys and Values come from another (encoder).

### IV. Attention Variants (Optional)

* **Flash Attention:** An IO-aware algorithm that speeds up attention and reduces memory usage.
* **Sparse Attention:** Only allows tokens to attend to a subset of others to save computation.
* **Sliding-window Attention:** Tokens only attend to their immediate neighbors.

---

## PART 3: Multi-Head Attention (Representation Diversity)

Instead of one "view," the model uses multiple "heads" to focus on different types of relationships simultaneously.

### I. Head Splitting

The d_{model} vector is split into h heads, each with dimension d_k = d_{model} / h.

### II. Parallel Attention Heads

Each head performs scaled dot-product attention independently, allowing the model to attend to syntax, grammar, and semantics separately.

### III. Head Concatenation

The outputs from all heads are concatenated back together and passed through a final linear layer.

---

## PART 4: Feed-Forward & Stabilization Block

Ensures the representations are refined and the training remains stable.

### I. Position-wise Feed-Forward Network (FFN)

Applied to each token independently. It usually consists of two linear transformations with an activation function in between:


### II. Activation Functions

* **ReLU:** The classic non-linear activation.
* **GELU (Gaussian Error Linear Unit):** Adds a stochastic element; standard in BERT and GPT.
* **SwiGLU:** A variant used in Llama models that combines Swish and Gated Linear Units.

### III. Residual Connections

Add the input of a layer back to its output (x + Sublayer(x)) to help gradients flow through deep networks.

### IV. Layer Normalization

* **Post-LayerNorm:** Normalization occurs *after* the residual addition (Original Transformer).
* **Pre-LayerNorm:** Normalization occurs *before* the sub-layer; results in much more stable training for large models.

---

## PART 5: Output Layer & Prediction

Translates the internal hidden states back into human-readable or task-specific data.

### I. Final Hidden Representation

The output of the last Transformer block, representing a contextualized understanding of the sequence.

### II. Output Projection

A linear layer that maps the d_{model} vector to the size of the total vocabulary.

### III. Softmax Normalization

Converts the raw scores (logits) into a probability distribution where all values sum to 1.

### IV. Decoding Strategies

* **Greedy Decoding:** Always picks the token with the highest probability.
* **Beam Search:** Explores multiple paths to find the most likely sequence.
* **Top-k / Top-p Sampling:** Introduces randomness by picking from the most likely tokens to make text more diverse and creative.

---