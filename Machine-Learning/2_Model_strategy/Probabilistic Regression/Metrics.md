# Probabilistic Regression Metrics

## Table of Contents

1. [Evaluate the Full Distribution](#1-evaluate-the-full-distribution)
   - [1.1 CRPS](#11-crps)
   - [1.2 Log Score / Negative Log-Likelihood](#12-log-score--negative-log-likelihood)
   - [1.3 Interval Coverage](#13-interval-coverage)
   - [1.4 Pinball Loss](#14-pinball-loss)
   - [1.5 WQL](#15-wql)
2. [Evaluate a Specific Threshold Event (Classification Metrics)](#2-evaluate-a-specific-threshold-event-classification-metrics)
   - [2.1 Brier Score](#21-brier-score)
   - [2.2 Precision / Recall or ROC-AUC](#22-precision--recall-or-roc-auc)
3. [Regression Metrics (Point Accuracy)](#3-regression-metrics-point-accuracy)

---

## 1. Evaluate the Full Distribution

These metrics assess how well the model captures the *shape and spread* of future values — not just the mean.

### 1.1 CRPS
*(Continuous Ranked Probability Score)*

### 1.2 Log Score / Negative Log-Likelihood

### 1.3 Interval Coverage

### 1.4 Pinball Loss

### 1.5 WQL
*(Weighted Quantile Loss)*

---

## 2. Evaluate a Specific Threshold Event (Classification Metrics)

These metrics treat threshold-exceedance as a binary classification problem.

### 2.1 Brier Score

Good for evaluating a single threshold event, but throws away magnitude once the binary label is created.

$$\text{Brier} = (p - y)^2$$

| Case | $p_i$ | $o_i$ | $(p_i - o_i)^2$ |
|------|------:|------:|----------------:|
| 1    |  0.90 |     1 |          0.0100 |
| 2    |  0.80 |     0 |          0.6400 |
| 5    |  0.55 |     0 |          0.3025 |
| 9    |  0.10 |     0 |          0.0100 |
| 10   |  0.05 |     1 |          0.9025 |

**Comparison with Log Loss:** Log Loss punishes confidently wrong predictions much more harshly than Brier Score.

$$\text{Log Loss} = -\left[y \log p + (1 - y) \log(1 - p)\right]$$

### 2.2 Precision / Recall or ROC-AUC

---

## 3. Regression Metrics (Point Accuracy)

Evaluates the point forecast (typically the predicted median) against the true value.

```python
metrics_point = {
    'MAE' : mae(y_act, y_med),
    'RMSE': rmse(y_act, y_med),
}
```
