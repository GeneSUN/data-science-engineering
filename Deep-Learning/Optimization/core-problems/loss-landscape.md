# The Loss Landscape

Neural network loss functions are non-convex — unlike convex problems (one global minimum), the landscape has many complex features that can trap or slow training.

---

### Local Minima

A point where all gradients are zero, but it's not the global minimum.

**Is this actually a problem?** Less than it sounds. Empirically, most local minima in deep networks have similar loss values to the global minimum — the network has enough capacity that many solutions work equally well. In high dimensions, truly bad local minima are rare.

---

### Saddle Points

A point where gradients are zero but it's a minimum in some directions and a maximum in others.

**This is the bigger problem.** In high-dimensional spaces, saddle points are far more common than local minima. Pure gradient descent can stall here because gradients near a saddle point are near zero.

**Solution:** Momentum helps — the accumulated velocity carries the optimizer through the flat saddle region. Adam is particularly effective.

---

### Plateaus & Flat Regions

Large regions where the loss changes very little. Gradients are small but nonzero — training doesn't diverge, but progress is extremely slow.

**Solutions:**
- **Learning rate scheduling** — higher LR to move through flat regions faster
- **Momentum** — builds speed across flat terrain
- **Adam** — adaptive LR prevents getting stuck when gradients are consistently small

---

### Sharp vs Flat Minima

Not all minima are equal. Sharp minima have steep walls — small perturbations cause large loss increases. Flat minima are broad and robust.

**Flat minima generalize better.** A model in a sharp minimum is fragile to slight data distribution shifts; a model in a flat minimum is robust.

**SGD with noise** (from mini-batches) tends to find flatter minima than full-batch gradient descent — one reason why pure batch GD is not always preferable.

---

### Summary

| Feature | Problem | Solution |
|---------|---------|---------|
| Local minima | Trap optimizer at suboptimal point | Usually not an issue in deep nets |
| Saddle points | Stall training (gradients ≈ 0) | Momentum, Adam |
| Plateaus | Very slow progress | LR scheduling, Adam |
| Sharp minima | Poor generalization | Mini-batch noise, weight decay |
