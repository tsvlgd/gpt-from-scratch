# GPT from Scratch — Decoder-Only Transformer (Character-Level)

A clean, from-scratch implementation of a decoder-only Transformer (GPT-style) language model in PyTorch, trained on Tiny Shakespeare at the character level.

This project focuses on understanding and implementing the core architecture of GPT, including masked self-attention, multi-head attention, residual connections, layer normalization, MLP blocks, weight tying, and autoregressive generation.

It is designed as a learning-oriented yet structurally organized implementation, inspired by:

* *Attention Is All You Need* (Vaswani et al., 2017)
* GPT / GPT-2 architecture (Radford et al.)
* Andrej Karpathy’s nanoGPT and lecture series

---

## 1. Architecture Overview

This implementation builds a **decoder-only Transformer** with:

* Token embeddings
* Learned positional embeddings
* Stacked Transformer blocks
* Multi-head masked self-attention
* Feedforward MLP with GELU
* Residual connections
* Pre-layer normalization
* Dropout regularization
* Weight tying between input embeddings and output projection
* Autoregressive generation with temperature and top-k sampling

### Model Configuration

```python
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int 
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True
```

This corresponds to a ~10M parameter GPT model when trained on Tiny Shakespeare.

---

## 2. Dataset

Character-level modeling on Tiny Shakespeare.

Dataset stats:

* Total characters (tokens): ~1.1M
* Vocabulary size: 65 unique characters
* Train split: 90%
* Validation split: 10%

Since this is character-level:

1 character = 1 token
No BPE or subword tokenization is used.

---

## 3. Transformer Block Structure

Each block contains:

1. Multi-Head Masked Self-Attention
2. Feedforward MLP (4× expansion)
3. Residual connections
4. Pre-LayerNorm

### Attention Head

* Linear projections for Q, K, V
* Causal mask via lower-triangular matrix
* Scaled dot-product attention
* Softmax + dropout
* Concatenation across heads
* Output projection

### MLP

* Linear: `n_embd → 4 * n_embd`
* GELU activation
* Linear: `4 * n_embd → n_embd`
* Dropout

---

## 4. Implemented Features

### Core Architecture

* [x] Decoder-only Transformer
* [x] Multi-head masked self-attention
* [x] Scaled dot-product attention
* [x] Residual connections
* [x] Pre-LayerNorm
* [x] GELU activation
* [x] Dropout
* [x] Weight tying
* [x] Custom GPT-style weight initialization (Normal(0, 0.02))

### Training

* [x] Cross-entropy loss
* [x] Batch sampling
* [x] Train/validation split
* [x] AdamW optimizer
* [x] Loss estimation loop

### Generation

* [x] Autoregressive decoding
* [x] Context cropping to block_size
* [x] Temperature scaling
* [x] Top-k sampling

---

## 5. Project Structure

```
.
├── data
│   └── shakespeare_char
│       └── input.txt
├── notebooks
│   ├── lm1.ipynb
│   └── lm2.ipynb
├── src
│   └── gpt_trainer
│       ├── data
│       │   └── prepare.py
│       ├── models
│       │   ├── bigram.py
│       │   ├── gpt.py
│       │   └── __init__.py
│       └── train.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Where to Look

* `models/gpt.py` → Full Transformer architecture
* `train.py` → Training loop and evaluation
* `data/prepare.py` → Dataset processing
* `bigram.py` → Minimal baseline language model

---

## 6. Parameter Estimation (~10M Model)

Given:

* n_layer = 6
* n_head = 6
* n_embd = 384
* vocab_size = 65
* block_size = 256

Approximate parameter count:

* Transformer blocks: ~10.6M
* Token embeddings: ~25K
* Positional embeddings: ~98K
* LM head: tied with embedding

Total ≈ **10.7M parameters**

Most parameters come from:

* MLP layers
* Attention projection matrices

---

## 7. Concepts Covered

This project reinforces:

* Autoregressive language modeling
* Causal masking
* Multi-head attention mechanics
* Parameter sharing (weight tying)
* Transformer scaling intuition
* Initialization strategy for deep networks
* Optimizer behavior (AdamW)
* Batch construction for language modeling
* Loss estimation for train/val

---

## 8. Roadmap / TODO (Towards nanoGPT Level)

### Architectural Upgrades

* [ ] Replace per-head Linear projections with single QKV projection
* [ ] Implement efficient `CausalSelfAttention` module
* [ ] Add Flash Attention support (PyTorch ≥ 2.0)
* [ ] Add optional bias-free LayerNorm

### Tokenization

* [ ] Implement BPE tokenizer
* [ ] Integrate `tiktoken`
* [ ] Compare char-level vs subword modeling

### Training Improvements

* [ ] Learning rate scheduler
* [ ] Gradient clipping
* [ ] Mixed precision training
* [ ] Fused AdamW
* [ ] Checkpoint saving and loading

### Scaling

* [ ] GPT-2 small configuration
* [ ] Parameter count benchmarking
* [ ] Throughput benchmarking

### Engineering

* [ ] Separate ModelConfig and TrainingConfig
* [ ] CLI interface for training
* [ ] Structured logging
* [ ] Model export

---

## 9. Key Insight

This implementation builds the conceptual foundation of GPT:

* Transformer decoder stack
* Causal masked attention
* Deep residual architecture
* Autoregressive token prediction

It prioritizes architectural understanding over production optimization.

---

## References

* Vaswani et al., *Attention Is All You Need* (2017)
* Karpathy, nanoGPT
* Karpathy, Neural Networks: Zero to Hero
* PyTorch Documentation
