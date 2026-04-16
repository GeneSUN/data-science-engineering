# Optimization in Deep Learning

Training a neural network is an optimization problem — minimize the loss over millions of parameters. But unlike convex problems with a clean global solution, deep learning optimization is messy, high-dimensional, and non-convex. This section covers why it's hard and how we handle it.

---

### 1. What are we actually optimizing?

Given a loss function $\mathcal{L}(W)$, we want to find weights $W$ that minimize it:

$$W^* = \arg\min_W \mathcal{L}(W)$$

Backpropagation gives us the gradients. Gradient descent uses them to take steps downhill. Everything else in this section is about making those steps reliable, efficient, and generalizable.

---

### 2. What makes this hard?

Four core failure modes:

| Problem | Description |
|---------|-------------|
| [Vanishing / Exploding Gradients](core-problems/vanishing-exploding-gradients.md) | Gradients shrink or blow up across layers — early layers stop learning or training diverges |
| [Loss Landscape](core-problems/loss-landscape.md) | Non-convex surface with saddle points, plateaus, and sharp minima |
| [Overfitting](core-problems/overfitting.md) | Model minimizes training loss but fails to generalize |
| [Hyperparameter Sensitivity](core-problems/hyperparameter-sensitivity.md) | Training outcome is fragile to learning rate, batch size, initialization |

---

### 3. What techniques do we use?

| Technique | What it solves |
|-----------|---------------|
| [Optimizers](techniques/optimizers.md) — SGD, Momentum, Adam | Faster, more stable convergence |
| [Normalization](techniques/normalization.md) — BatchNorm, LayerNorm | Stabilizes activations, tames vanishing gradients |
| [Initialization](techniques/initialization.md) — Xavier, He | Prevents gradient issues from the first forward pass |
| [LR Scheduling](techniques/lr-scheduling.md) — warmup, cosine decay | Adapts step size across training phases |
| [Regularization](techniques/regularization.md) — Dropout, Weight Decay | Closes the gap between training and generalization |

---

### 4. How do the pieces fit together?

A typical training setup uses all of these in combination:

```
He init → BatchNorm → Adam optimizer → cosine LR schedule
                                      + weight decay + dropout
```

No single technique solves everything. Gradient stability, convergence speed, and generalization are separate concerns — each addressed by a different layer of the stack.
