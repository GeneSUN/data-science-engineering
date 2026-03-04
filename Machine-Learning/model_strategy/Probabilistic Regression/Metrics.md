## Table of Contents

1. [Evaluate the full distribution](#1-evaluate-the-full-distribution)
   - [CRPS](#11-crps)
   - [Log score / Negative log-likelihood](#12-log-score--negative-log-likelihood)
   - [Interval Coverage](#13-interval_coverage)
   - [Pinball Loss](#14-pinball_loss)
2. [Evaluate the specific threshold event (Classification Metrics)](#2-evaluate-the-specific-threshold-event-classification-metrics)
   - [Brier Score](#21-brier-score)
   - [Precision / Recall or ROC-AUC](#22-precisionrecall-or-roc-auc)
3. [Regression Metrics](#3-regression-metrics)

## 1. Evaluate the full distribution

### 1.1. CRPS

### 1.2. Log score / Negative log-likelihood:

### 1.3. Interval_coverage

### 1.4. pinball_loss

## 2. Evaluate the specific threshold event-Classification Metrics

### 2.1. Brier score:

**Brier score: good for one threshold event, but throws away magnitude once the label is created.**

$$Brier=(p−y)^2$$

| Case | (p_i) | (o_i) | ((p_i-o_i)^2) |
| ---- | ----: | ----: | ------------: |
| 1    |  0.90 |     1 |          0.01 |
| 2    |  0.80 |     0 |          0.64 |
| 5    |  0.55 |     0 |        0.3025 |
| 9    |  0.10 |     0 |          0.01 |
| 10   |  0.05 |     1 |        0.9025 |


> this is very similar to cross-entropy, which punishes extreme confident wrong predictions much more harshly than Brier score.

$$LogLoss=−[ylogp+(1−y)log(1−p)]$$



### 2.2. Precision/Recall or ROC-AUC:


## 3. Regression Metrics

```python
metrics_point = {
    'MAE' : mae(y_act, y_med),
    'RMSE': rmse(y_act, y_med),
}
```






