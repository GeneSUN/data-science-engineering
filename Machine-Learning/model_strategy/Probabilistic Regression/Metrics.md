

## 1. Evaluate the full distribution
### CRPS

### Log score / Negative log-likelihood:

## 2. Evaluate the specific threshold event

### Brier score:

**Brier score: good for one threshold event, but throws away magnitude once the label is created.**

$$Brier=(p−y)^2$$

| Case | (p_i) | (o_i) | ((p_i-o_i)^2) |
| ---- | ----: | ----: | ------------: |
| 1    |  0.90 |     1 |          0.01 |
| 2    |  0.80 |     0 |          0.64 |
| 3    |  0.70 |     1 |          0.09 |
| 4    |  0.60 |     1 |          0.16 |
| 5    |  0.55 |     0 |        0.3025 |
| 6    |  0.40 |     0 |          0.16 |
| 7    |  0.30 |     1 |          0.49 |
| 8    |  0.20 |     0 |          0.04 |
| 9    |  0.10 |     0 |          0.01 |
| 10   |  0.05 |     1 |        0.9025 |


> this is very similar to cross-entropy, which punishes extreme confident wrong predictions much more harshly than Brier score.

$$LogLoss=−[ylogp+(1−y)log(1−p)]$$



### Precision/Recall or ROC-AUC:

