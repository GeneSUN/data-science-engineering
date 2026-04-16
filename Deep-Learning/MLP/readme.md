# A Dialectical Introduction to Deep Learning


### 1. What is the simplest deep learning model?

A **Multi-Layer Perceptron (MLP)** — a sequence of layers stacked together:

```
Input Layer → Hidden Layer(s) → Output Layer
```

Each layer transforms the signal from the previous one. The "deep" in deep learning simply means there are multiple hidden layers.

---

### 2. What happens inside each layer? 

Each neuron computes:

$$\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

- **W** (weight matrix): what the layer pays attention to
- **b** (bias): shifts the output
- **σ** (activation function): the nonlinearity

**Why not just stack linear layers — why do we need the activation function?**

A composition of linear transformations is still just a linear transformation. 
activation introduce non-linearity here.


---

### 3. Now we have a model. How does it learn?

Like any supervised model (e.g. linear regression), we define a **loss function** that measures how wrong the predictions are. The goal is to minimize it by adjusting **W** and **b**.

The challenge: with many layers, how do we know how much each weight contributed to the error?

**Backpropagation** solves this — it applies the chain rule of calculus to propagate gradients from the output back through each layer, giving us the gradient of the loss with respect to every weight.

**Gradient descent** then uses those gradients to update the weights:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \nabla_{\mathbf{W}} \mathcal{L}$$

Optimizers like **SGD** and **Adam** are strategies for doing this update more efficiently (e.g. using momentum, adaptive learning rates).

> Backpropagation computes the gradients. Gradient descent uses them. They are distinct steps.

---

### 4. bias-variance tradeoff

This is the **bias-variance tradeoff** — a too-simple model underfits, a too-complex one memorizes noise.

Common regularization techniques in deep learning:

- **Dropout**: randomly zeros out neurons during training, preventing co-adaptation
- **Batch Normalization**: normalizes activations within a mini-batch, stabilizing training and acting as mild regularization
- **Weight Decay (L2)**: penalizes large weights, keeping the model from over-relying on any one feature
- **Early Stopping**: halt training when validation loss stops improving

These don't change the model architecture — they shape how it learns, nudging it toward solutions that generalize.
