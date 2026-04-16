# Overfitting (Optimization vs Generalization Gap)

Minimizing training loss is not the actual goal — we want the model to perform well on unseen data. These two objectives can diverge.

---

### The Problem

A sufficiently large model can memorize the training set entirely, achieving near-zero training loss while performing poorly on new data. It learns noise and spurious patterns alongside the true signal.

This is the **bias-variance tradeoff**:
- **High bias (underfitting):** Model is too simple to capture the pattern
- **High variance (overfitting):** Model is too flexible and captures noise

---

### Why Deep Networks Are Particularly Prone

Deep networks have millions of parameters — far more than training samples in many settings. Classical learning theory says this should cause severe overfitting. In practice it doesn't always, but the risk is real.

---

### Solutions

**Reduce model capacity:**
- Use a smaller network (fewer layers / neurons)

**Add noise during training:**
- **Dropout** — randomly disables neurons, prevents co-adaptation
- **Data augmentation** — artificially expand the training set with transformations

**Penalize complexity:**
- **Weight Decay (L2)** — keeps weights small, discourages over-reliance on any feature
- **L1 regularization** — encourages sparse weights

**Stop before overfitting:**
- **Early stopping** — monitor validation loss, halt when it stops improving

**Stabilize training:**
- **BatchNorm / LayerNorm** — acts as mild regularization

---

### Diagnosing It

| Observation | Likely Cause |
|-------------|-------------|
| Training loss low, validation loss high | Overfitting |
| Both losses high | Underfitting |
| Both losses low | Good fit |
| Validation loss initially decreases then rises | Classic overfitting curve — use early stopping here |
