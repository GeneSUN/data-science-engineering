# Normalization

As data flows through layers, activations can grow or shrink uncontrollably — making training unstable. Normalization keeps activations in a reasonable range at each layer.

---

### Batch Normalization (BatchNorm)

Normalizes across the **batch dimension** for each feature:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

Then rescales with learned parameters $\gamma, \beta$:

$$y = \gamma \hat{x} + \beta$$

- $\mu_B, \sigma_B$: mean and variance computed over the current mini-batch
- At inference, uses running statistics from training

**Effect:** Stabilizes training, allows higher learning rates, acts as mild regularization.  
**Limitation:** Depends on batch size — breaks down with very small batches.

---

### Layer Normalization (LayerNorm)

Normalizes across the **feature dimension** for each sample:

$$\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

- $\mu_L, \sigma_L$: mean and variance computed over all features of a single sample

**Effect:** Same stabilizing benefit as BatchNorm, but independent of batch size.  
**Where it's used:** Transformers, LLMs, RNNs — anywhere batch statistics are unreliable.

---

### BatchNorm vs LayerNorm

| | BatchNorm | LayerNorm |
|--|-----------|-----------|
| Normalizes over | Batch (per feature) | Features (per sample) |
| Batch-size sensitive | Yes | No |
| Typical use | CNNs, MLPs | Transformers, RNNs |
