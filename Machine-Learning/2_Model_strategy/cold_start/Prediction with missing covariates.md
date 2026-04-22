# Prediction with Missing Covariates

## The Problem

Training rows have $(x_1, \ldots, x_n)$, but a new case arrives with only a subset $(x_1, \ldots, x_{n-p})$. The missing features $x_{n-p+1}, \ldots, x_n$ are unavailable at inference time.

---

## Approaches

### Option 0 — Rebuild a Smaller Model (Baseline)

Train a separate model using only $(x_1, \ldots, x_{n-p})$.

- **Pro:** Simple, no assumptions about the missing features
- **Con:** Discards all signal from $(x_{n-p+1}, \ldots, x_n)$ that was available at training time — see [distillation.md](distillation.md) for a way to recover this signal

---

### Option 1 — Treat Missing Features as Missing Values

Pass the partial input to the original model and let it handle the gaps natively.

- **Tree-based models** (XGBoost, LightGBM, CatBoost) handle missing values at split time — they learn the best default direction during training
- **Neural networks** can use masking or learned missing-value embeddings

**When it works well:** When missingness at inference matches the missingness pattern seen during training (i.e., the model has already learned to route around those features).

---

### Option 2 — Impute the Missing Features

Estimate the missing values before passing to the model.

#### Simple Imputation
- Mean/median/mode imputation — fast but ignores correlations
- k-NN imputation — uses similar complete rows to fill gaps

#### Multiple Imputation
Generate multiple plausible values for each missing feature, run predictions for each, and average the outputs. This propagates uncertainty from the missing features through to the final prediction.

---

### Option 3 — Bayesian Treatment (Most Principled)

Treat missing features as **latent variables** and integrate them out:

$$P(Y \mid x_1, \ldots, x_{n-p}) = \int P(Y \mid x_1, \ldots, x_n) \, P(x_{n-p+1}, \ldots, x_n \mid x_1, \ldots, x_{n-p}) \, dx$$

In practice:
- **PyMC** supports auto-imputation of missing covariates during MCMC sampling
- You define a joint model over both observed and missing features; the sampler infers the missing values alongside the prediction
- The output is a *distribution* over $Y$ that honestly reflects uncertainty from the missing inputs

**Advantage for cold start:** The model naturally outputs wider uncertainty when features are missing — no separate calibration step needed.
