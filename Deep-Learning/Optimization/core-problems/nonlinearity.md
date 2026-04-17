# Non-Linearity

Activation functions like ReLU make individual neurons nonlinear. But real-world data — images, language, signals — has structure far more complex than any single nonlinear function can capture. The deeper question is: how do we build networks capable of modeling that complexity?

Three structural answers, beyond just choosing an activation function:

---

### 1. Depth — Compositional Non-Linearity

Stacking layers compounds simple transformations into complex ones:

```
Layer 1: edges
Layer 2: shapes
Layer 3: objects
```

Each layer applies a simple nonlinear transformation. Composed together, they build hierarchical representations — early layers detect low-level patterns, later layers combine them into high-level concepts.

The "deep" in deep learning is doing real work here. Depth is what separates a shallow function approximator from a model capable of understanding structure.

---

### 2. Architecture Design — Structured Non-Linearity

Different architectures bake nonlinearity into the structure itself, not just the activation:

**Convolutions (CNNs)**  
Detect local patterns via shared filters. The same filter applied across the input creates position-invariant, structured nonlinear feature detection — something a flat MLP cannot do efficiently.

**Attention (Transformers)**  
Computes relationships between all elements dynamically — the weighting is *input-dependent*. This is stronger than static MLP transformations because the nonlinearity adapts to the specific input at inference time.

**Residual Connections (ResNet)**  
Rather than forcing each layer to learn a full transformation, the network learns incremental changes: $y = F(x) + x$. This makes it easier to compose many nonlinear layers without gradient degradation.

---

### 3. Gating Mechanisms — Conditional Non-Linearity

Used in RNNs, LSTMs, and Transformers. Gates learn *when* to apply a transformation:

```
if condition → apply transformation A
else         → apply transformation B
```

This creates piecewise nonlinear behavior — the network can route information differently depending on the input, rather than applying the same function uniformly. It's a form of learned conditional computation.

---

### Summary

| Approach | Mechanism | Where |
|----------|-----------|-------|
| Depth | Compose simple transformations into complex ones | All deep networks |
| Architecture | Bake structured nonlinearity into the design | CNNs, Transformers, ResNet |
| Gating | Learn conditional computation paths | LSTM, GRU, Transformers |
