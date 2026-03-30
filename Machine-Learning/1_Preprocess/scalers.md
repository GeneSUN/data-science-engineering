# Feature Scaling in Machine Learning: Standardization vs. Normalization
 
## Standardization and Feature Scaling

Many **distance-metric-based** and **gradient-based** learning algorithms implicitly assume that input features are on comparable scales—typically centered around zero with similar variances.

- Algorithms such as **K-Nearest Neighbors (KNN)**, **K-Means**, and **SVMs with RBF kernels** are highly sensitive to feature scale, since distances or inner products directly determine model behavior.
- A notable [exception](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)  is **tree-based models** (e.g., Decision Trees, Random Forests, Gradient Boosted Trees), which split data based on feature thresholds and are therefore largely **invariant to monotonic feature scaling**.
- In **deep learning**, proper feature scaling is especially important:
  - Activation functions such as **sigmoid** and **tanh** saturate for large positive or negative inputs, leading to vanishing gradients.
  - Optimization algorithms like **(stochastic) gradient descent** converge faster and more stably when inputs are scaled to ranges such as **[0, 1]** or **[-1, 1]**.
  - A classic example is image data, where raw pixel values in **[0, 255]** are typically normalized before training.

---

### Normalization (Min-Max Scaling)

**Min-max normalization** rescales features to a fixed range, commonly **[0, 1]**:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

This approach preserves relative distances but is **highly sensitive to outliers**, since extreme values directly affect the scaling range.  
As a result, normalization works best when:
- The feature has a bounded range
- Outliers are rare or already handled

For example, **age** is often a good candidate for min-max scaling due to its relatively bounded and uniform distribution, whereas **income** may not be ideal because a small number of high-income individuals can dominate the scaling.

---

### Standardization (Z-Score Scaling)

**Standardization** transforms features to have zero mean and unit variance:


$$x' = \frac{x - \mu}{\sigma}$$


This makes standardization particularly effective for:
- Distance-based models
- Linear models and regularization-based methods
- Gradient-based optimization algorithms
  
**When Min-Max Scaling Is Better Than Standardization**
- When You Want to Preserve Sparsity or Zero Meaning
- Bounded Features With Known Limits, Neural Networks



### Question:
**When during the data preprocessing pipeline should standardization be applied?** 
Standardization should be applied after splitting the data into training and testing sets to prevent data leakage. The standardization parameters (mean and standard deviation) should be calculated on the training data and then applied to both the training and testing sets to ensure consistency.

