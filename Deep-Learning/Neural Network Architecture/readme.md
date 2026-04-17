# Neural Network Architectures

MLP is the simplest deep learning structure — but different data types and problems demand different architectural designs. This section covers the major architecture families and the design patterns that enhance them.

---

## Architecture Families

Each family is built around a different assumption about the structure of the data:

| Architecture | Built for | Key idea |
|-------------|-----------|---------|
| **MLP** | Tabular / general | Fully connected layers, no structural assumption |
| **CNN** | Images / spatial data | Local patterns, translation invariance via convolution |
| **RNN / LSTM / GRU** | Sequences / time series | Hidden state carries memory across timesteps |
| **Transformer** | Sequences (language, vision) | Attention over all positions simultaneously |

---

## Design Patterns

Enhancements that cut across architecture families — most modern networks combine several of these:

| Pattern | What it solves |
|---------|---------------|
| **Residual connections** | Vanishing gradients in deep networks — creates a gradient highway |
| **Attention** | Static transformations can't model dynamic relationships — attention weights inputs by relevance |
| **Normalization** | Activations drift across layers — BatchNorm / LayerNorm keeps signal stable |

---

## Articles

- [Attention](Attention.md)
- [RNN-based architectures](RNN-based.md)
