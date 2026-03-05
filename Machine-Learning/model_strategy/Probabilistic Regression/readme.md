# Table of Contents

1. [Derived Probabilistic Prediction Methods](#derived-probabilistic-prediction-methods)
    - [1. Prediction intervals using conformal prediction](#1-prediction-intervals-using-conformal-prediction)
    - [2. Binary classification on a thresholded target](#2-binary-classification-on-a-thresholded-target)
2. [Probabilistic Modeling Approaches](#probabilistic-modeling-approaches)
    - [Category 1 — Compatible Probabilistic Models](#category-1--compatible-probabilistic-models-modify-classical-mldl-models)
        - [1.1 Quantile Regression Models](#11-quantile-regression-models)
        - [1.2 Parametric Neural Networks](#12-parametric-neural-networks)
    - [Category 2 — Specialized Probabilistic Models](#category-2--specialized-probabilistic-models)
        - Gaussian Processes
        - Diffusion-Based Probabilistic Models
        - [Why NGBoost is not popular?](#why-ngboost-is-not-popular)

#  Derived Probabilistic Prediction Methods

## 1. Prediction intervals using conformal prediction

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

## 2. Binary classification on a thresholded target

Convert a continuous target **Y** into an event indicator $`Z_u = \mathbb{1}\{Y > u\}`$, then model  $$Pr(Y>u∣X)$$

- what Regression does:

$$Y_{pred}​=E[Y∣X]$$

- Convert regression to exceeding classification:

$$Pr(Y>u∣X)$$

Example
- [AWS predictive maintenance: predict failure probability, then trigger action if it exceeds a threshold](https://aws.amazon.com/blogs/iot/asset-maintenance-with-aws-iot-services-predict-and-respond-to-potential-failures-before-they-impact-your-business/)

---

# Probabilistic Modeling Approaches:

## Category 1 — Compatible Probabilistic Models (Modify classical ML/DL models)

These models were not originally designed for probabilistic prediction, but can be adapted.

### 1.1 Quantile Regression Models

Instead of minimizing MSE: 

$$\{min\} E[(Y−Y^2)]$$

Use quantile loss:

$$
L_{\tau}(y,\hat{y}) =
\begin{cases}
\tau (y-\hat{y}) & \text{if } y > \hat{y} \\
(1-\tau)(\hat{y}-y) & \text{if } y \le \hat{y}
\end{cases}
$$

Example: XGBoost/LightGBM/Random Forest/Neural networks

```python
# -------------------------
# 3) Train quantile models (q10, q50, q90)
# -------------------------

quantiles = [0.1, 0.5, 0.9]
models = {}

for q in quantiles:
    model = XGBRegressor(
        objective="reg:quantileerror",  # quantile (pinball) loss :contentReference[oaicite:4]{index=4}
        quantile_alpha=q,               # which quantile to estimate :contentReference[oaicite:5]{index=5}
        **base_params
    )
    # Early stopping keeps it efficient
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    models[q] = model

# Predict
pred_q10 = models[0.1].predict(X_test)
pred_q50 = models[0.5].predict(X_test)
pred_q90 = models[0.9].predict(X_test)

# -------------------------
# 4) Evaluate critical metrics
# -------------------------
metrics = {}

for q, pred in [(0.1, pred_q10), (0.5, pred_q50), (0.9, pred_q90)]:
    metrics[f"pinball_loss_q{int(q*100)}"] = pinball_loss(y_test, pred, q)

# Optional: RMSE on median (not the main objective, but useful sanity check)
metrics["rmse_q50"] = mean_squared_error(y_test, pred_q50)

# Interval metrics from (q10, q90)
metrics["interval_coverage_80pct_(q10,q90)"] = interval_coverage(y_test, pred_q10, pred_q90)
metrics["mean_interval_width_(q90-q10)"] = float(np.mean(pred_q90 - pred_q10))
metrics["crossing_rate_q10>q50"] = quantile_crossing_rate(pred_q10, pred_q50)
metrics["crossing_rate_q50>q90"] = quantile_crossing_rate(pred_q50, pred_q90)
metrics["crossing_rate_q10>q90"] = quantile_crossing_rate(pred_q10, pred_q90)
```

### 1.2 Parametric Neural Networks

PyTorch
- What it does: You implement it by outputting **distribution parameters** and using **NLL loss**.
- Key modules:
  - torch.distributions (Normal, LogNormal, Gamma, etc.)
  - Loss pattern: loss = -dist.log_prob(y).mean()


```python
mu, sigma = self.forward(x)
dist = torch.distributions.Normal(mu, sigma)
loss = -dist.log_prob(y).mean()
```

## Category 2 — Specialized Probabilistic Models

These models are designed from the ground up to estimate probability distributions.

$$Y∣X∼Dist(θ(X))$$

| Model Type                          | What it gives                  | Suitable for                     |
| ----------------------------------- | ------------------------------ | -------------------------------- |
| Gaussian Processes                  |     $f(x)∼GP(m(x),k(x,x′))$  |                                    |
| **NGBoost**                         | natural gradients to optimize probabilistic loss functions  | Best off-the-shelf |
| **Diffusion-Based Probabilistic Models**  | Treeffuser    | Very strong uncertainty modeling |


**why NGBoost is not popular?**
1. NGBoost is fundamentally based on sequential boosting with probabilistic gradients, which makes scaling harder.
2. XGBoost/LightGBM/CatBoost/PyTorch/TensorFlow, They have good Engineering Ecosystem.

If you look at the tech industry:
1. Quantile Models Solve 80% of Problems, Most real probabilistic business questions are simple.
2. The remainning is Dominated by Deep Learning, in Large-Scale Probabilistic Systems
## 1. Prediction intervals using conformal prediction
