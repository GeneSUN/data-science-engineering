Classical Regression:

$$Y_pred​=E[Y∣X]$$

$$Prob(Y>n∣X)$$

To do that, your model must produce a distribution estimate, not just a single point.

Modeling Approaches:

## 1. Distributional Regression (Preferred for this task)

Model predicts full conditional distribution:

$$Y∣X∼Dist(θ(X))$$

| Model Type                          | What it gives                  | Suitable for                     |
| ----------------------------------- | ------------------------------ | -------------------------------- |
| **Gaussian / Lognormal regression** | Mean + variance → distribution | Simple baseline                  |
| **NGBoost**                         | Full probabilistic prediction  | Best off-the-shelf               |
| **Parametric Bayesian regression**  | Posterior distribution of Y    | Very strong uncertainty modeling |

P(Y>n)=1−F(n∣θ(X))

## 2. conformal prediction

PyTorch
- What it does: You implement it by outputting distribution parameters and using NLL loss.
- Key modules:
  - torch.distributions (Normal, LogNormal, Gamma, etc.)
  - Loss pattern: loss = -dist.log_prob(y).mean()

Pyro (probabilistic programming on PyTorch)
- What it does: Full probabilistic modeling + Bayesian inference (more powerful, more complex)
- Package: pyro-ppl

Quantile Regression
- Predicts quantiles (P50/P90/etc.), not a full parametric distribution

Libraries:
- lightgbm (quantile objective)
- xgboost (quantile in newer versions; depends on your install)
- statsmodels (QuantReg)
- sklearn (GradientBoostingRegressor has quantile loss; also HistGradientBoosting in some setups)


Gaussian Processes (GP)
