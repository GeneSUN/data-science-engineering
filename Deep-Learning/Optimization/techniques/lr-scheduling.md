# Learning Rate Scheduling

A fixed learning rate is rarely optimal throughout training. Too high early on causes instability; too high late in training prevents convergence to a good minimum.

Schedulers adjust $\eta$ automatically over the course of training.

---

### Warmup

Start with a very small learning rate and ramp up over the first few epochs.

**Why:** At initialization, weights are random and gradients are unreliable. A large early step can send training into a bad region it never recovers from. Warmup buys time for the model to stabilize.

Common in Transformers — typically warm up for 4–10% of total training steps.

---

### Step Decay

Reduce learning rate by a fixed factor every N epochs:

$$\eta \leftarrow \eta \times \gamma \quad \text{every } k \text{ epochs}$$

Simple and interpretable. Common default: drop by $0.1$ every 30 epochs.

---

### Cosine Annealing

Decay learning rate following a cosine curve from $\eta_{max}$ to $\eta_{min}$:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{t\pi}{T}\right)$$

Smooth decay that slows naturally near the end of training. Widely used in vision and NLP.

---

### Cosine Annealing with Warm Restarts (SGDR)

Periodically reset the learning rate back to $\eta_{max}$ after each cosine cycle. Each restart lets the optimizer explore a different region of the loss landscape.

---

### Comparison

| Schedule | Shape | Typical Use |
|----------|-------|-------------|
| Warmup | Ramp up | Transformer training (combined with other schedules) |
| Step Decay | Staircase down | CNNs, classic training recipes |
| Cosine Annealing | Smooth curve down | General — widely adopted default |
| Cosine + Warm Restarts | Repeated cosine cycles | When exploring multiple minima |
