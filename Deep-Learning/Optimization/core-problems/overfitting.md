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

**1. Lower accuracy — prevent the model from fitting too precisely**

| Technique | How it helps |
|-----------|-------------|
| Reduce model capacity | Fewer layers / neurons limits the model's ability to memorise |
| Early stopping | Halt training when validation loss stops improving |

**2. Introduce error — inject noise so the model cannot rely on exact patterns**

| Technique | How it helps |
|-----------|-------------|
| Data augmentation | Expands the training set with transformations; model sees more variation |
| Dropout | Randomly disables neurons each step; prevents co-adaptation |
| L2 regularization (weight decay) | Penalises large weights; discourages over-reliance on any single feature |
| L1 regularization | Penalises weight magnitude; encourages sparse weights |

---

### Diagnosing It

| Observation | Likely Cause |
|-------------|-------------|
| Training loss low, validation loss high | Overfitting |
| Both losses high | Underfitting |
| Both losses low | Good fit |
| Validation loss initially decreases then rises | Classic overfitting curve — use early stopping here |
