# GP is “a probability distribution over functions”


The kernel k(x,x') is the definition of similarity: if two inputs are “similar,” the degree of their outputs correlated.
- close in time → strongly correlated
- far apart → weakly correlated

**Why ```uncertainty``` behaves nicely**
- Near many observations → uncertainty shrinks
- Far from data → uncertainty grows back toward the prior

This “mean + uncertainty band” behavior is a big reason people like GPs.

## Math

<img width="592" height="714" alt="Screenshot 2026-03-01 at 4 28 18 PM" src="https://github.com/user-attachments/assets/17c22803-3cd5-4235-b554-85bd27a95f97" />


## Procedure
<img width="789" height="1796" alt="output (8)" src="https://github.com/user-attachments/assets/c13e0086-7828-4454-b8f4-c13b9f31c537" />

### Step 1. Before seeing data, you define a prior

In GPR, you first specify what kinds of functions you think are plausible before looking at the training data.
This prior has two parts:
- a mean function
  - 0 if ```normalize_y=False``` or the training target mean if ```normalize_y=True```
- a covariance function, which is the kernel
  - prior covariance is specified by the kernel object you pass in

### Step 2. The kernel 

- RBF kernel → smooth functions
- Periodic kernel → repeating patterns
- DotProduct kernel → linear-style behavior

1. The kernel has hyperparameters;
2. during fitting, GPR learns those hyperparameters**
3. the kernel hyperparameters are optimized by maximizing the log-marginal-likelihood (LML).

> “Find the smoothness and noise level that best explain the data without overcomplicating the function.”


### Step 3. Noise is handled by alpha or WhiteKernel

- Option A: alpha
- Option B: WhiteKernel

### Step 4. After fitting, GP predicts a mean and standard deviation


### Summary
$$Posterior ∝ Prior×Likelihood$$


## Practical
**When GP is a great choice**
- You have small/medium data
- You care about uncertainty
- You want a strong nonlinear model without deep learning
- You want to encode domain beliefs via kernels (smooth/periodic/etc.)

**When GP struggles**
- Big data (classic GP needs an 𝑛×𝑛 kernel matrix and matrix inversion → expensive)
- Very high-dimensional inputs (sometimes works, sometimes not)

