# Weight Initialization

Before training starts, weights must be initialized. This matters more than it sounds — bad initialization causes gradients to vanish or explode before training even begins.

**Why not initialize to zero?** All neurons compute identical gradients and learn the same thing. The network never breaks symmetry.

**Why not initialize too large?** Activations saturate (Sigmoid, Tanh hit their flat regions), gradients vanish.

---

### Xavier Initialization (Glorot)

Designed for **Sigmoid / Tanh** activations. Keeps variance stable across layers:

$$W \sim \mathcal{U}\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$

Scales weights by the number of input and output connections, so signal neither grows nor shrinks as it passes through.

---

### He Initialization (Kaiming)

Designed for **ReLU** activations. ReLU kills half its inputs (sets negatives to zero), so we need larger initial weights to compensate:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

The factor of 2 accounts for the roughly half of neurons that ReLU zeros out.

---

### Which to use

| Activation | Initialization |
|-----------|---------------|
| ReLU / Leaky ReLU | He (Kaiming) |
| Sigmoid / Tanh | Xavier (Glorot) |
| Linear | Xavier |

In PyTorch, He initialization is applied automatically when you use `nn.Linear` with ReLU.
