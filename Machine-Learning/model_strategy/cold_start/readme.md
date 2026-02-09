## 0) Make generalization possible: remove “ID memorization”

If the destination is new, an ID feature can’t help. **The model needs destination attributes available at day-0:**

- geo/region/country, distance bands, time zone offsets
- customs / cross-border flags, carrier coverage, service level tiers
- infrastructure proxies (warehouse type, last-mile density, port/airport proximity)

Onsite phrasing: **“For cold start, I’d ensure features are transferable (available for unseen markets), not purely historical aggregates keyed by destination ID.”**

---

**Bayesian Hierarchical Modeling**
- https://bayesball.github.io/BOOK/bayesian-hierarchical-modeling.html

**Predict a distribution, not just a mean (so exceedance is easy + uncertainty is honest)**


The advantage in cold start is that the model can naturally output wider uncertainty when the input is out-of-distribution or data-sparse.
[Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation]

**Handle distribution shift explicitly (new market often = covariate shift)**


**Similarity-based priors: “find similar markets/lanes” and shrink to them**











