# Transformer Attention Mechanisms

This repository provides a detailed overview of **attention mechanisms in Transformers**, including self-attention, cross-attention, multi-head attention, and efficient attention types. The visual diagram below summarizes all attention types and how they operate in Transformer architectures.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Attention Types](#attention-types)
   - [Self-Attention](#self-attention)
   - [Masked Self-Attention](#masked-self-attention)
   - [Cross-Attention](#cross-attention)
   - [Multi-Head Attention](#multi-head-attention)
   - [Local / Windowed Attention](#local--windowed-attention)
   - [Global / Sparse Attention](#global--sparse-attention)
   - [Linear / Efficient Attention](#linear--efficient-attention)
   - [Relative / Rotary Positional Attention](#relative--rotary-positional-attention)
3. [Attention Complexity Table](#attention-complexity-table)
4. [Visual Diagram](#visual-diagram)
5. [References](#references)

---

## Introduction
Transformers use **attention mechanisms** to capture relationships between tokens in a sequence. Each attention type is designed to solve specific problems, such as handling long sequences, enabling autoregressive generation, or efficiently encoding positions.

---

## Attention Types

### 1. Self-Attention
- Allows each token to attend to all other tokens in the same sequence.
- Formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- Captures global context for all tokens.

### 2. Masked Self-Attention
- Used in **autoregressive decoding**.
- Prevents attending to future tokens.

### 3. Cross-Attention
- Used in **encoder-decoder models**.
- Decoder queries attend to encoder outputs.
- Queries (Q) from decoder, Keys (K) & Values (V) from encoder.

### 4. Multi-Head Attention
- Multiple attention heads capture different relationships.
- Each head has separate Q, K, V projections.
- Outputs concatenated and linearly projected.

### 5. Local / Windowed Attention
- Each token attends only to a local window of tokens.
- Reduces computational complexity: O(n·w) instead of O(n²).

### 6. Global / Sparse Attention
- Certain tokens (like CLS) attend globally, others attend locally.
- Efficient for long sequences.

### 7. Linear / Efficient Attention
- Uses linear approximations to reduce memory and computation.
- Examples: Linformer, Performer.

### 8. Relative / Rotary Positional Attention
- Encodes **distance between tokens** dynamically.
- Rotary attention rotates queries and keys in embedding space.

---

## Attention Complexity Table

| Attention Type                  | Purpose / Use Case                     | Complexity |
|---------------------------------|--------------------------------------|------------|
| Self-Attention                  | Contextualizing each token            | O(n²)      |
| Masked Self-Attention           | Autoregressive decoding               | O(n²)      |
| Cross-Attention                 | Decoder attends to encoder            | O(n²)      |
| Multi-Head Attention            | Capture multiple relationships        | O(h·n²)    |
| Local / Windowed Attention      | Long sequences, local context         | O(n·w)     |
| Global / Sparse Attention       | Efficient long-range context          | O(n·log n) |
| Linear / Efficient Attention    | Memory-efficient long sequences       | O(n)       |
| Relative / Rotary Positional    | Encode positional info efficiently   | O(n²)      |

---

## Visual Diagram

```mermaid
flowchart TB
    %% Input
    Input["Input Sequence Tokens"] --> Embedding["Token Embedding + Positional Encoding"]

    %% Self-Attention
    Embedding --> SelfAttention["Self-Attention"]
    SelfAttention --> MaskedSelfAttention["Masked Self-Attention"]

    %% Multi-Head Attention
    SelfAttention --> MultiHead["Multi-Head Attention"]
    MaskedSelfAttention --> MultiHead

    %% Cross-Attention
    MultiHead --> CrossAttention["Cross-Attention - Decoder attends Encoder"]

    %% Efficient / Long Sequence Attention
    CrossAttention --> LocalAttention["Local - Windowed Attention"]
    CrossAttention --> GlobalAttention["Global - Sparse Attention"]
    CrossAttention --> LinearAttention["Linear - Efficient Attention"]

    %% Positional
    LocalAttention --> Positional["Relative - Rotary Positional Attention"]
    GlobalAttention --> Positional
    LinearAttention --> Positional

    %% Final Representation
    Positional --> FinalHidden["Final Hidden Representation"]

    %% Linear Projection to logits
    FinalHidden --> LinearProj["Linear Projection W^O"]
    LinearProj --> Logits["Logits"]
    
    %% Softmax
    Logits --> Softmax["Softmax Normalization"]
    Softmax --> Prob["Probability Distribution over Vocabulary"]
    
    %% Decoding Strategies
    Prob --> Greedy["Greedy Decoding"]
    Prob --> Beam["Beam Search"]
    Prob --> TopK["Top-k / Top-p Sampling"]

    %% Table of Complexity
    subgraph "Attention Complexity Table"
        SA["Self-Attention: O(n²) - Contextualizing each token"]
        MSA["Masked Self-Attention: O(n²) - Autoregressive decoding"]
        CA["Cross-Attention: O(n²) - Decoder attends to encoder"]
        MHA["Multi-Head Attention: O(h·n²) - Capture multiple relationships"]
        LA["Local / Windowed: O(n·w) - Long sequences, local context"]
        GA["Global / Sparse: O(n·log n) - Efficient long-range context"]
        LE["Linear / Efficient: O(n) - Memory-efficient long sequences"]
        RP["Relative / Rotary: O(n²) - Encode positional info efficiently"]
    end

    SelfAttention --> SA
    MaskedSelfAttention --> MSA
    CrossAttention --> CA
    MultiHead --> MHA
    LocalAttention --> LA
    GlobalAttention --> GA
    LinearAttention --> LE
    Positional --> RP

