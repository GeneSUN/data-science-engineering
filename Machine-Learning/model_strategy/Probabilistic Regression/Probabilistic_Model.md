
## 1.Prediction intervals using conformal prediction

Conformal prediction (1) uses a regression model to produce point forecasts, and (2) uses **empirical forecast errors** to construct prediction **intervals**.

```
Regression model → point forecast
              +
Empirical residual quantiles → interval width
              =
Conformal prediction interval
```

For example:
| True y | Predicted y | Residual |
| ------ | ----------- | -------- |
| 1000   | 980         | 20       |
| 1100   | 1050        | 50       |
| 900    | 950         | -50      |

- If you use variance of residual, $$s^2 = Σ(r_i - r̄)^2 / (n-1)$$
  
  - For ŷ=980  -> [930, 1030]
  - For ŷ=1050 -> [1000, 1100]
  - For ŷ=950  -> [900, 1000]
  
- If you want the usual 90% interval (α = 0.10), q = 50:

## 2.Binary classification on a thresholded target

Convert a continuous target **Y** into an event indicator $`Z_u = \mathbb{1}\{Y > u\}`$, then model  $$Pr(Y>u∣X)$$

- what Regression does:

$$Y_{pred}​=E[Y∣X]$$

- Convert regression to exceeding classification:

$$Pr(Y>u∣X)$$

Example
- [AWS predictive maintenance: predict failure probability, then trigger action if it exceeds a threshold](https://aws.amazon.com/blogs/iot/asset-maintenance-with-aws-iot-services-predict-and-respond-to-potential-failures-before-they-impact-your-business/)
- 

Modeling Approaches:

## 1. Quantile Regression 

- ```reg:quantileerror```: Quantile loss, also known as ```pinball loss```. See later sections for its parameter and Quantile Regression for a worked example.

## 2.  Loss function: Negative Log Likelyhood

PyTorch
- What it does: You implement it by outputting distribution parameters and using NLL loss.
- Key modules:
  - torch.distributions (Normal, LogNormal, Gamma, etc.)
  - Loss pattern: loss = -dist.log_prob(y).mean()


```python
mu, sigma = self.forward(x)
dist = torch.distributions.Normal(mu, sigma)
loss = -dist.log_prob(y).mean()
```

## 3. Distributional Regression (Preferred for this task)

Model predicts full conditional distribution:

$$Y∣X∼Dist(θ(X))$$

| Model Type                          | What it gives                  | Suitable for                     |
| ----------------------------------- | ------------------------------ | -------------------------------- |
| **Gaussian / Lognormal regression** | Mean + variance → distribution | Simple baseline                  |
| Gaussian Processes                  |                                |                                  |
| **NGBoost**                         | Full probabilistic prediction  | Best off-the-shelf               |
| **Parametric Bayesian regression**  | Posterior distribution of Y    | Very strong uncertainty modeling |

P(Y>n)=1−F(n∣θ(X))



Pyro (probabilistic programming on PyTorch)
- What it does: Full probabilistic modeling + Bayesian inference (more powerful, more complex)
- Package: pyro-ppl

