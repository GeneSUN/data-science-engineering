# Regularization

A model that minimizes training loss perfectly often fails on new data — it memorized the training set instead of learning the underlying pattern. Regularization constrains the model during training to improve generalization.

---

### Dropout

Randomly zero out a fraction $p$ of neurons during each forward pass:

- Forces the network to not rely on any single neuron
- Effectively trains an ensemble of sub-networks
- At inference, all neurons are active but scaled by $(1-p)$

Typical $p = 0.1$–$0.5$. Higher dropout = stronger regularization.  
**Where:** Dense layers in MLPs, Transformers. Less common in CNNs (BatchNorm fills this role).

---

### Weight Decay (L2 Regularization)

Adds a penalty on large weights to the loss:

$$\mathcal{L}_{total} = \mathcal{L} + \lambda \sum_i W_i^2$$

During the gradient update, this shrinks every weight slightly toward zero at each step:

$$W \leftarrow W(1 - \eta\lambda) - \eta \nabla_W \mathcal{L}$$

Prevents over-reliance on any single feature. Typical $\lambda = 10^{-4}$–$10^{-2}$.

> In Adam, "weight decay" and "L2 regularization" behave differently — use **AdamW** (decoupled weight decay) for correct behavior.

---

### Early Stopping

Monitor validation loss during training. Stop when it stops improving.

- No hyperparameter to tune
- Cheap: just requires saving a checkpoint at the best validation step
- Pairs well with other regularization — acts as a final safety net

**Practical tip:** Use a patience window (e.g. stop after 10 epochs without improvement) rather than stopping at the exact minimum.

---

### Summary

| Technique | Mechanism | Typical Use |
|-----------|-----------|-------------|
| Dropout | Randomly disable neurons | MLPs, Transformers |
| Weight Decay | Penalize large weights | Universally applicable |
| Early Stopping | Halt at best validation point | Always worth doing |
