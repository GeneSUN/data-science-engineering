# Vanishing & Exploding Gradients

During backpropagation, gradients are multiplied layer by layer via the chain rule. In a deep network, this means the gradient at layer 1 is the product of many Jacobians:

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial W_n} \cdot \prod_{k=2}^{n} \frac{\partial h_k}{\partial h_{k-1}}$$

If each factor is < 1, the product shrinks exponentially → **vanishing gradients** (early layers stop learning).  
If each factor is > 1, the product grows exponentially → **exploding gradients** (training diverges).

---

### Vanishing Gradients

**Cause:** Sigmoid and Tanh saturate — their derivatives approach 0 at large inputs. Multiplied across many layers, the gradient reaches zero.

**Symptoms:** Early layers have near-zero gradients and learn almost nothing. Loss plateaus early.

**Solutions:**
- Use **ReLU** activations (derivative is 1 for positive inputs — no saturation)
- Use **residual connections** (skip connections add a gradient highway that bypasses layers)
- Use **careful initialization** (Xavier / He — prevents activations from entering saturation zones at the start)
- Use **BatchNorm / LayerNorm** (keeps activations in well-behaved ranges)

---

### Exploding Gradients

**Cause:** Large weights or activations cause gradients to compound rapidly backward.

**Symptoms:** Loss spikes to NaN, weights become very large, training becomes unstable.

**Solutions:**
- **Gradient clipping** — cap the gradient norm before each update:
  $$g \leftarrow g \cdot \frac{\text{threshold}}{\|g\|}  \quad \text{if } \|g\| > \text{threshold}$$
- **Careful initialization** — prevents large activations from the start
- **Normalization** — stabilizes activations layer by layer

---

### Summary

| Problem | Root Cause | Key Fix |
|---------|-----------|---------|
| Vanishing | Saturating activations, deep chains | ReLU, ResNet skip connections, LayerNorm |
| Exploding | Large gradient products | Gradient clipping, careful initialization |
