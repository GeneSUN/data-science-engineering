# Hyperparameter Sensitivity

Unlike model parameters (weights), hyperparameters are set before training and not learned. Wrong choices can make a good architecture fail entirely.

---

### Learning Rate

The most important hyperparameter. Controls the step size of each gradient update.

- **Too high:** Loss oscillates or diverges
- **Too low:** Training converges extremely slowly, may get stuck

**Solutions:**
- Use **learning rate scheduling** (warmup + decay)
- Use **Adam** — its adaptive LR reduces sensitivity compared to SGD
- Use **LR range test** — sweep LR over a few iterations to find a stable range

---

### Weight Initialization

Random initialization that's too large or too small causes vanishing/exploding gradients before training even begins.

**Solutions:**
- **He initialization** for ReLU
- **Xavier initialization** for Sigmoid/Tanh
- These are now default in most frameworks — rarely needs manual tuning

---

### Batch Size

Affects both training speed and generalization quality.

- **Large batches:** More accurate gradients, faster wall-clock time, but tend toward sharp minima
- **Small batches:** Noisier gradients, but noise acts as implicit regularization toward flatter minima

**Common default:** 32–256 for most tasks. For large models (LLMs), much larger batches with linear LR scaling.

---

### Architecture Choices

Depth, width, skip connections — these interact with all of the above. A deeper network needs more careful initialization and normalization.

---

### Practical Advice

You rarely need to tune everything. The usual priority:

1. **Learning rate** — biggest impact, tune first
2. **Batch size** — adjust based on hardware and target quality
3. **Weight decay** — small but consistent effect on generalization
4. **Architecture depth/width** — tune last, after training is stable
