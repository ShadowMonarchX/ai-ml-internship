# Types of Transformers in Artificial Intelligence

## Overview

This repository is a **structured, in-depth reference** to the major **types of Transformer architectures** used in modern Artificial Intelligence. Transformers form the backbone of today’s state-of-the-art systems in **natural language processing, computer vision, speech, and multimodal AI**, enabling scalable learning through attention-based architectures.

This document is designed for:
- Students and researchers
- Engineers and practitioners
- Interview and exam preparation
- Architecture-level understanding (from theory to practice)

---

## Table of Contents

- [Introduction](#introduction)
- [Core Transformer Architecture](#core-transformer-architecture)
- [Key Papers (Must-Reads)](#key-papers-must-reads)
- [High-Level Classification Table](#high-level-classification-table)
- [Attention Variants](#attention-variants)
- [Training & Alignment Concepts](#training--alignment-concepts)
- [Transformer Architectures — Type-wise Breakdown](#transformer-architectures--type-wise-breakdown)
- [Glossary of Terms](#glossary-of-terms)
- [Future Directions](#future-directions)
- [Learning Resources](#learning-resources)
- [License](#license)

---

## Introduction

Transformers are a class of neural network architectures specifically designed for **sequence modeling and representation learning**. Unlike RNNs or CNNs, Transformers rely on **self-attention mechanisms** to model global dependencies, enabling:
- Parallel computation
- Efficient scaling
- Strong long-range context modeling

These properties make Transformers ideal for large-scale AI systems.

---

## Core Transformer Architecture

The original Transformer architecture (encoder–decoder) is built from the following core components:

- **Token / Patch Embeddings** – Convert raw inputs into dense vectors
- **Positional Encoding** – Inject sequence or spatial order information
- **Multi-Head Attention (MHA)** – Attend to information from multiple representation subspaces
- **Feed-Forward Networks (FFN)** – Apply non-linear transformations per position
- **Residual Connections & Layer Normalization** – Enable deep and stable training

---

## Key Papers (Must-Reads)

To understand the evolution of Transformers, the following papers are foundational:

| Paper Title | Year | Significance |
| --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Introduced the original Transformer architecture. |
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | 2018 | Popularized Encoder-only models and Masked Language Modeling. |
| [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 2020 | Demonstrated the power of scaling Decoder-only models. |
| [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | 2021 | Applied Transformer architecture successfully to Computer Vision. |
| [Exploring the Limits of Transfer Learning (T5)](https://arxiv.org/abs/1910.10683) | 2019 | Unified NLP tasks into a text-to-text format using Encoder-Decoder. |
| [Switch Transformers (MoE)](https://arxiv.org/abs/2101.03961) | 2021 | Scaled models to trillions of parameters using Mixture of Experts. |


---

## High-Level Classification Table

| Type | Main Use | Representative Models |
|-----|---------|-----------------------|
| Encoder-only | Understanding / Extraction | BERT, RoBERTa, ELECTRA |
| Decoder-only | Generative AI / Chat | GPT-4, Llama 3, Mistral |
| Encoder–Decoder | Translation / Summarization | T5, BART, FLAN-T5 |
| Vision (ViT) | Image Analysis | ViT, Swin, MAE |
| Multimodal | Vision + Language | CLIP, Flamingo, Gemini, Claude 3 |
| Efficient | Long Contexts | FlashAttention, Longformer, BigBird |
| MoE | Large-scale Scaling | Mixtral 8x7B, Switch Transformer |

---

## Attention Variants

- **Self-Attention**
- **Masked (Causal) Attention**
- **Cross-Attention**
- **FlashAttention**
- **Sparse Attention**
- **Sliding-Window Attention**

---

## Training & Alignment Concepts

1. **Pre-training** (Self-supervised learning)
2. **Supervised Fine-Tuning (SFT)**
3. **RLHF (Reinforcement Learning from Human Feedback)**
4. **DPO (Direct Preference Optimization)**

---

# Transformer Architectures — Type-wise Breakdown

## ENCODER-ONLY TRANSFORMERS (BERT, RoBERTa, ELECTRA)

### PART 1: Input Representation & Encoding
I. Tokenization  
II. Token Embedding  
&nbsp;&nbsp;a. Learned embeddings  
III. Positional Information  
&nbsp;&nbsp;a. Sinusoidal positional encoding  
&nbsp;&nbsp;b. Learned positional embedding  
IV. Input Composition  

### PART 2: Attention Mechanism (Context Formation)
I. Query–Key–Value Projection  
II. Scaled Dot-Product Attention  
III. Self-Attention  

### PART 3: Multi-Head Attention
I. Head Splitting  
II. Parallel Attention Heads  
III. Head Concatenation  

### PART 4: Feed-Forward & Stabilization
I. Position-wise Feed-Forward Network  
II. GELU Activation  
III. Residual Connections  
IV. Post-LayerNorm  

### PART 5: Output Layer
I. Final Hidden Representation  
II. Output Projection  
III. Classification Head  

---

## DECODER-ONLY TRANSFORMERS (GPT-4, Llama 3, Mistral)

### PART 1: Input Representation & Encoding
I. Tokenization  
II. Token Embedding  
&nbsp;&nbsp;a. Learned  
&nbsp;&nbsp;b. Shared / tied  
III. Positional Information  
&nbsp;&nbsp;a. RoPE  
&nbsp;&nbsp;b. ALiBi  
IV. Input Composition  

### PART 2: Attention Mechanism
I. Query–Key–Value Projection  
II. Scaled Dot-Product Attention  
III. Masked Self-Attention  

### PART 3: Multi-Head Attention
I. Head Splitting  
II. Parallel Attention Heads  
III. Head Concatenation  

### PART 4: Feed-Forward & Stabilization
I. Position-wise Feed-Forward Network  
II. SwiGLU Activation  
III. Residual Connections  
IV. Pre-LayerNorm  

### PART 5: Output Layer
I. Final Hidden Representation  
II. Output Projection  
III. Next-Token Prediction  
IV. Decoding Strategies  

---

## ENCODER–DECODER TRANSFORMERS (T5, BART, FLAN-T5)

### PART 1: Input Encoding
I. Source Tokenization  
II. Token Embedding  
III. Positional Encoding  

### PART 2: Attention
I. Encoder Self-Attention  
II. Decoder Masked Self-Attention  
III. Cross-Attention  

### PART 3: Multi-Head Attention
I. Head Splitting  
II. Parallel Heads  
III. Concatenation  

### PART 4: Feed-Forward & Stabilization
I. Feed-Forward Network  
II. ReLU / GELU  
III. Residuals & LayerNorm  

### PART 5: Output
I. Decoder Representation  
II. Sequence-to-Sequence Output  

---

## VISION TRANSFORMERS (ViT, Swin, MAE)

### PART 1: Input Representation
I. Image Patch Extraction  
II. Patch Embedding  
III. Learned Positional Encoding  

### PART 2–5
Self-Attention → Multi-Head Attention → Feed-Forward → Classification Head

---

## MULTIMODAL TRANSFORMERS (CLIP, Flamingo, Gemini, Claude 3)

### PART 1: Multi-Modal Encoding
I. Text Encoding  
II. Vision Encoding  
III. Modality Embedding  

### PART 2: Attention
I. Intra-Modal Attention  
II. Cross-Modal Attention  

### PART 3–5
Multi-Head Attention → Feed-Forward → Unified Output Head

---

## EFFICIENT TRANSFORMERS (FlashAttention, Longformer, BigBird)

### PART 1: Encoding
I. Tokenization  
II. Embedding  
III. ALiBi  

### PART 2: Efficient Attention
I. Flash Attention  
II. Sparse / Sliding Attention  

### PART 3–5
Efficient MHA → FFN → Output Projection

---

## MIXTURE OF EXPERTS (MoE) TRANSFORMERS (Mixtral, Switch)

### PART 1: Encoding
I. Tokenization  
II. Embedding  

### PART 2–3
Self-Attention → Multi-Head Attention  

### PART 4: Expert Layer
I. Routing Network  
II. Sparse Expert Selection  
III. Expert FFNs  

### PART 5: Output
I. Final Representation  
II. Output Projection  

---

## Glossary of Terms

- **Token** – Basic processing unit
- **Context Window** – Maximum visible tokens
- **Parameters** – Trainable weights
- **Hallucination** – Confident but incorrect output
- **Zero/Few-Shot** – Learning with minimal examples

---

## Future Directions

- Infinite / Long Context Transformers
- On-device & Quantized Models
- World Models & Robotics
- Neuro-Symbolic Transformers

---
## Learning Resources

* **Visualizations:** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar.
* **Code:** [Hugging Face Course](https://huggingface.co/learn/nlp-course/) for practical implementation.
* **Mathematical Depth:** [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/).
---

## License

MIT License
