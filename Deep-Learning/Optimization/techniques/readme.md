# Optimization Techniques

These techniques form a coherent story — each one addressing a specific problem at a specific stage of training.

```
[1. Initialize] → [2. Forward Pass] → [3. Compute Loss] → [4. Backprop + Update] → repeat
                       ↑                                          ↑
               BatchNorm, Dropout                        Optimizer, LR Schedule
```

---

<details>
<summary><b>Step 1 — Initialization</b></summary>

Before training begins, weights must be set carefully. Bad initialization causes gradients to vanish or explode before the first update even happens.

→ [He / Xavier Initialization](initialization.md)

</details>

<details>
<summary><b>Step 2 — Forward Pass</b></summary>

As activations flow layer to layer, two problems can emerge:

- **Activations drift** → [Normalization](normalization.md) (BatchNorm / LayerNorm)
- **Overfitting** → [Regularization](regularization.md) (Dropout, Weight Decay)
- **Network too deep to learn complex nonlinear patterns** → [Skip Connections](skip-connections.md)

</details>

<details>
<summary><b>Step 3 — Weight Update</b></summary>

After backprop computes the gradients, two choices shape how weights are updated:

- **How to step** → [Optimizer](optimizers.md) (SGD, Adam)
- **How large to step** → [LR Scheduling](lr-scheduling.md) (warmup, cosine decay)

</details>

<details>
<summary><b>Step 4 — When to Stop</b></summary>

Training longer isn't always better — at some point validation loss starts rising.

→ [Early Stopping](regularization.md)

</details>
