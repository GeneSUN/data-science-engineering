# Optimizers

All optimizers share the same goal: use the gradient $\nabla_W \mathcal{L}$ to update weights. They differ in *how* that step is computed.

---

### SGD (Vanilla)

$$W \leftarrow W - \eta \cdot \nabla_W \mathcal{L}$$

Subtract the gradient, scaled by learning rate $\eta$. Simple, but sensitive to learning rate choice and slow in ravines.

---

### SGD + Momentum

$$v \leftarrow \beta v + \nabla_W \mathcal{L}$$
$$W \leftarrow W - \eta \cdot v$$

Accumulates a running average of past gradients. Builds up speed in consistent directions, dampens oscillation.
Typical $\beta = 0.9$.

---

### RMSProp

$$s \leftarrow \beta s + (1-\beta)(\nabla_W \mathcal{L})^2$$
$$W \leftarrow W - \frac{\eta}{\sqrt{s + \epsilon}} \cdot \nabla_W \mathcal{L}$$

Adapts the learning rate per-weight by dividing by the root of recent squared gradients. Weights that move a lot get a smaller step.

---

### Adam (Adaptive Moment Estimation)

Combines momentum + RMSProp:

$$m \leftarrow \beta_1 m + (1-\beta_1)\nabla_W \mathcal{L} \quad \text{(1st moment)}$$
$$v \leftarrow \beta_2 v + (1-\beta_2)(\nabla_W \mathcal{L})^2 \quad \text{(2nd moment)}$$
$$W \leftarrow W - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

- **m**: tracks direction (like momentum)
- **v**: tracks magnitude (like RMSProp)
- $\hat{m}, \hat{v}$: bias-corrected estimates for early training

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$.

**Adam is the default choice** for most deep learning tasks — robust, fast, requires little tuning.

---

### Comparison

| Optimizer | Adapts LR? | Momentum? | Best for |
|-----------|-----------|-----------|---------|
| SGD | No | No | Simple, interpretable baselines |
| SGD + Momentum | No | Yes | CV with careful tuning |
| RMSProp | Yes | No | RNNs, non-stationary problems |
| Adam | Yes | Yes | General default |

> **Note:** "SGD" in frameworks like PyTorch (`torch.optim.SGD`) refers to the *update rule*, not the batch strategy. Mini-batch is almost always used underneath.
