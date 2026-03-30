# GP is “a probability distribution over functions”

# Gaussian Process (GP) — Quick Guide

## Core Idea
A **Gaussian Process (GP)** is a **distribution over functions**.

Instead of learning one function, GP models **many possible functions + uncertainty**.

---

## Bayesian View

### 1. Prior — belief before data

Define what functions look like:

- Mean function:  
  - usually 0 (or data mean if normalized)

- Kernel (covariance):  
  defines similarity → correlation of outputs

Examples:
- RBF → smooth
- Periodic → repeating
- DotProduct → linear

<img width="689" height="1796" alt="output (8)" src="https://github.com/user-attachments/assets/c13e0086-7828-4454-b8f4-c13b9f31c537" />


**Interpretation**  
> “What kinds of functions do I expect?”

---

### 2. Likelihood — data assumption

Assume noisy observations:

$$
y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

Noise handling:
- `alpha` (fixed noise)
- `WhiteKernel` (learn noise)

**Interpretation**  
> “If this function is true, how likely is the data?”

---

### 3. Posterior — updated belief

Combine prior + data:

$$
\text{Posterior} \propto \text{Prior} \times \text{Likelihood}
$$

Output:
- mean prediction
- uncertainty (std)

Key behavior:
- near data → low uncertainty
- far from data → high uncertainty

**Interpretation**  
> “Updated distribution of functions after seeing data”

---

## Learning

GP learns **kernel hyperparameters** by maximizing:

- Log Marginal Likelihood (LML)

Goal:
> fit data well + avoid overfitting

---

## Summary

```text
Prior     → define function shape (kernel)
Likelihood→ assume noise model
Posterior → update with data → mean + uncertainty
```



## Practical
**When GP is a great choice**
- You have small/medium data
- You care about uncertainty
- You want a strong nonlinear model without deep learning
- You want to encode domain beliefs via kernels (smooth/periodic/etc.)

**When GP struggles**
- Big data (classic GP needs an 𝑛×𝑛 kernel matrix and matrix inversion → expensive)
- Very high-dimensional inputs (sometimes works, sometimes not)

