# Cold Start Problem

Cold start occurs when a model must make predictions for entities it has never seen before — new markets, new destinations, new users — where no historical signal exists.

---

## The Core Challenge

Models trained on historical data tend to rely on **ID-level memorization** (e.g., destination ID embeddings, user history aggregates). When a new entity arrives, those features are unavailable and the model degrades silently.

**Principle:** For cold start, ensure features are *transferable* — available for unseen entities at day-0 — not purely historical aggregates keyed by an ID.

**Examples of transferable features:**
- Geo/region/country, distance bands, time zone offsets
- Customs / cross-border flags, carrier coverage, service level tiers
- Infrastructure proxies (warehouse type, last-mile density, port/airport proximity)

---

## Strategy Overview

### 1. Feature Engineering — Remove ID Memorization
Design features that generalize to unseen entities from day-0. Replace ID-keyed aggregates with structural/attribute-based features that are always available.

### 2. Similarity-Based Priors — "Find Similar Entities"
When a new entity arrives, find the most similar *warm* entities and shrink predictions toward them. This is the frequentist analogue of a Bayesian prior.

- Embed entities by their attributes, find k-nearest neighbors in that space
- Use neighbor predictions as a prior; update as new data accumulates

### 3. Bayesian Hierarchical Modeling
Model each entity as a draw from a population distribution. New entities inherit the population prior and update as data arrives.

- Natural fit for cold start: uncertainty is wide at day-0 and narrows with observations
- Reference: https://bayesball.github.io/BOOK/bayesian-hierarchical-modeling.html

### 4. Predict a Distribution, Not Just a Mean
A probabilistic output gives honest uncertainty at cold start — the model naturally outputs wider intervals when inputs are out-of-distribution or data-sparse.

- Enables exceedance probability estimation without post-hoc calibration
- Reference: *Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation*

### 5. Handle Distribution Shift Explicitly
A new market is often a covariate shift problem — the feature distribution at inference differs from training. Techniques:

- Domain adaptation (reweight training samples toward the new distribution)
- Conformal prediction with covariate-shift correction

### 6. Knowledge Distillation — Cold Start via Reduced Feature Set
Train a *teacher* on full features (warm data) and distill its knowledge into a *student* that operates on only the cold-start feature subset.

→ See [distillation.md](distillation.md)

### 7. Prediction with Missing Covariates
When a new case arrives with only a partial feature set, treat missing features as latent variables rather than rebuilding a smaller model from scratch.

→ See [Prediction with missing covariates.md](Prediction%20with%20missing%20covariates.md)
