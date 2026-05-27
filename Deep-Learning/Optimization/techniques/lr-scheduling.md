# Learning Rate Scheduling

A fixed learning rate is rarely optimal throughout training. Too high early on causes instability; too high late in training prevents convergence to a good minimum.

Schedulers adjust $\eta$ automatically over the course of training.

---

### Warmup

Start with a very small learning rate and ramp up over the first few epochs.

**Why:** At initialization, weights are random and gradients are unreliable. A large early step can send training into a bad region it never recovers from. Warmup buys time for the model to stabilize.

Common in Transformers — typically warm up for 4–10% of total training steps.

```python
import torch
from torch.optim.lr_scheduler import LambdaLR

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
warmup_steps = 100

def warmup_fn(step):
    if step < warmup_steps:
        return step / warmup_steps  # ramp from 0 → 1
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)

for step in range(total_steps):
    optimizer.step()
    scheduler.step()
```

---

### Step Decay

Reduce learning rate by a fixed factor every N epochs:

$$\eta \leftarrow \eta \times \gamma \quad \text{every } k \text{ epochs}$$

Simple and interpretable. Common default: drop by $0.1$ every 30 epochs.

```python
from torch.optim.lr_scheduler import StepLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

---

### Cosine Annealing

Decay learning rate following a cosine curve from $\eta_{max}$ to $\eta_{min}$:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{t\pi}{T}\right)$$

Smooth decay that slows naturally near the end of training. Widely used in vision and NLP.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

---

### Cosine Annealing with Warm Restarts (SGDR)

Periodically reset the learning rate back to $\eta_{max}$ after each cosine cycle. Each restart lets the optimizer explore a different region of the loss landscape.

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# T_0: steps in first cycle; T_mult: multiply cycle length after each restart
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

---

### Comparison

| Schedule | Shape | Typical Use |
|----------|-------|-------------|
| Warmup | Ramp up | Transformer training (combined with other schedules) |
| Step Decay | Staircase down | CNNs, classic training recipes |
| Cosine Annealing | Smooth curve down | General — widely adopted default |
| Cosine + Warm Restarts | Repeated cosine cycles | When exploring multiple minima |
