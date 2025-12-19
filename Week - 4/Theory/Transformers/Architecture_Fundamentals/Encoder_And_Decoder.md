# Transformer Encoder and Decoder Explained (With Full Theory & Mathematics)

Transformers have revolutionized Natural Language Processing (NLP) by replacing recurrence with attention mechanisms.  
At the core of the Transformer architecture are two fundamental components:

- **Encoder** – responsible for understanding input data  
- **Decoder** – responsible for generating output data  

This README provides a **complete theoretical and mathematical explanation** of:
- Encoder
- Decoder
- Encoder vs Decoder architectures

No unrelated topics are included.

---

## What Is a Transformer?

A **Transformer** is a deep learning architecture introduced in  
**“Attention Is All You Need” (Vaswani et al., 2017)**.

Unlike RNNs or LSTMs, Transformers:
- Use **self-attention** instead of recurrence
- Process entire sequences **in parallel**
- Scale efficiently to very large datasets

A Transformer can be built using:
- Encoder blocks only
- Decoder blocks only
- Encoder + Decoder blocks together

---

## Mathematical Notation Used

- Sequence length: `n`
- Model dimension: `d_model`
- Input tokens: `x1, x2, ..., xn`
- Embedding matrix: `E`
- Positional encoding: `P`
- Query, Key, Value dimensions: `d_k`, `d_v`

---

## Encoder: Understanding the Input

The **Encoder** converts an input sequence into **context-aware vector representations** using **bi-directional self-attention**.

Encoders see **all tokens at once**, allowing full contextual understanding.

---

## Encoder Workflow (Theory + Math)

### 1. Input Tokenization

Example:
```

"The quick brown fox"
→ ["The", "quick", "brown", "fox"]

```

---

### 2. Token Embedding

Each token is mapped to a dense vector:

```

X = [x1, x2, ..., xn]
xi ∈ R^(d_model)

```

Using an embedding matrix:

```

xi = E(token_i)

```

---

### 3. Positional Encoding

Transformers have no inherent notion of order, so positional encodings are added:

```

Z0 = X + P

```

**Sinusoidal Positional Encoding:**

```

PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

```

Why this works:
- Encodes absolute & relative positions
- Allows extrapolation to longer sequences
- No extra learned parameters

---

### 4. Multi-Head Self-Attention (Core Mechanism)

For each token representation `Z`:

```

Q = Z * W_Q
K = Z * W_K
V = Z * W_V

```

Where:
- `Q` = Queries
- `K` = Keys
- `V` = Values

---

### Scaled Dot-Product Attention

```

Attention(Q, K, V) = softmax( (Q × Kᵀ) / sqrt(d_k) ) × V

```

Key points:
- `Q × Kᵀ` computes similarity
- `sqrt(d_k)` prevents large gradients
- `softmax` converts scores into probabilities

---

### 5. Multi-Head Attention

Instead of a single attention head:

```

head_i = Attention(Q_i, K_i, V_i)

```

Combine multiple heads:

```

MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) × W_O

```

Why multi-head?
- Captures multiple relationships
- Learns syntax, semantics, and structure simultaneously

---

### 6. Add & Layer Normalization

Residual connection + normalization:

```

Z1 = LayerNorm( Z0 + MultiHead(Z0) )

```

Purpose:
- Preserve original information
- Improve gradient flow
- Stabilize training

---

### 7. Feed-Forward Neural Network (FFN)

Applied independently to each token:

```

FFN(x) = max(0, x × W1 + b1) × W2 + b2

```

- First layer expands dimension
- ReLU adds non-linearity
- Second layer projects back to `d_model`

---

### 8. Final Encoder Output

```

Z_out = LayerNorm( Z1 + FFN(Z1) )

```

This produces **contextual embeddings** for each token.

---

## Encoder Output Usage

- Used directly for prediction tasks
- Or passed to the decoder

---

## Encoder-Only Models

Used for **understanding tasks**:

- BERT
- RoBERTa
- DistilBERT
- Vision Transformer (ViT)

---

## Decoder: Generating the Output

The **Decoder** generates output sequences **autoregressively**, one token at a time.

Each token depends **only on past tokens**.

---

## Decoder Workflow (Theory + Math)

### 1. Input Embedding + Positional Encoding

```

Y0 = E(y_<t) + P

```

Where:
- `y_<t` = previously generated tokens

---

### 2. Masked Multi-Head Self-Attention

Causal mask definition:

```

Mask[i][j] = 0       if j ≤ i
Mask[i][j] = -∞      if j > i

```

Masked attention formula:

```

Attention = softmax( (Q × Kᵀ) / sqrt(d_k) + Mask ) × V

```

Why masking?
- Prevents seeing future tokens
- Ensures left-to-right generation

---

### 3. Encoder–Decoder Attention

Queries from decoder, keys & values from encoder:

```

Attention(Q_dec, K_enc, V_enc)

```

Purpose:
- Align input sequence with output sequence
- Critical for translation & summarization

---

### 4. Feed-Forward Network

Same FFN as encoder:

```

FFN(x) = max(0, x × W1 + b1) × W2 + b2

```

---

### 5. Output Projection + Softmax

Logits computation:

```

logits = Z × W_vocab

```

Probability distribution:

```

P(token_t) = softmax(logits)

```

---

## Decoder Output

- Generates one token at a time
- Stops at `<EOS>` or max length

---

## Decoder-Only Models

Used for **generation tasks**:

- GPT
- GPT-2
- GPT-3
- GPT-4

---

## Encoder vs Decoder (Mathematical Comparison)

| Feature | Encoder | Decoder |
|------|--------|--------|
| Attention | Full self-attention | Masked self-attention |
| Direction | Bi-directional | Uni-directional |
| Uses Mask | No | Yes |
| Output Type | Embeddings | Probabilities |
| Objective | Understanding | Generation |
| Training | Parallel | Autoregressive |

---

## Encoder–Decoder Together (Seq2Seq)

Overall flow:

```

Input X → Encoder → Hidden States H → Decoder → Output Y

```

Used in:
- Machine translation
- Summarization
- Question answering

Models:
- Transformer
- T5
- BART

---

## Real-World Analogy

**Live Translation Booth**

- Encoder → Listener fully understanding the sentence
- Decoder → Speaker generating translation word-by-word

---

## Key Concepts to Remember

- Self-attention replaces recurrence
- Scaling factor stabilizes gradients
- Masking enforces causality
- FFN adds non-linearity
- Residual connections preserve information
- Softmax produces probabilities

---

## Conclusion

The Encoder and Decoder form the **mathematical and conceptual foundation** of Transformer models.  
Understanding their **functions, equations, and roles** is essential for mastering modern NLP and generative AI.