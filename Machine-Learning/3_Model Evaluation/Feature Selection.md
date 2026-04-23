# Feature Engineering

<details>
<summary>1. Domain Knowledge (Start Here)</summary>

Before applying any statistical or ML technique:
- Consult business context and subject matter experts
- Understand which features are likely relevant or redundant
- Identify data quality issues, potential leakage features, and proxy variables

</details>

<details>
<summary>2. Statistical Analysis</summary>

<details>
<summary>2a. EDA</summary>

- Distribution of features
- Remove features with low variance
- Flag features with too many missing values

</details>

<details>
<summary>2b. Feature–Target Correlation</summary>

- **Numerical target** — Pearson or Spearman correlation coefficients
- **Categorical target** — Chi-squared test or Mutual Information

</details>

<details>
<summary>2c. Multicollinearity Check</summary>

- Compute correlation matrix or Variance Inflation Factor (VIF) to detect highly correlated features
- Drop or combine redundant variables

</details>

</details>

<details>
<summary>3. ML-Based Feature Selection</summary>

<details>
<summary>3a. Subset Selection</summary>

- Forward Selection, Backward Elimination, or Recursive Feature Elimination (RFE)

</details>

<details>
<summary>3b. Regularization Methods</summary>

- **Lasso (L1)** or **ElasticNet** automatically shrink unimportant features
- Especially useful in high-dimensional data

</details>

<details>
<summary>3c. Dimensionality Reduction</summary>

- **PCA** or **Truncated SVD** for unsupervised feature compression
- Useful when dealing with highly correlated numeric variables

</details>

</details>

<details>
<summary>4. Post-Modeling Feature Selection</summary>

<details>
<summary>4a. Model-Based Feature Importance</summary>

- Tree-based models (Random Forest, XGBoost) provide native importance scores
- Use **SHAP values** or **permutation importance** for more interpretable results

</details>

<details>
<summary>4b. Statistical Significance (linear models)</summary>

- Use p-values from GLM or logistic regression to assess feature significance
- Beware: multicollinearity and sample size can distort p-values

</details>

</details>

<details>
<summary>5. Collaboration and Feedback</summary>

Present selected features and their importance to:
- **Business stakeholders** — validate whether features make domain sense
- **Engineering / domain experts** — check for technical relevance

Use this feedback loop to adjust, refine, or re-engineer features.

</details>

<details>
<summary>6. Model-Specific Considerations</summary>

Different models have different feature sensitivities:

| Model Type | Feature Sensitivity |
|---|---|
| **KNN** | Sensitive to irrelevant or high-dimensional features — needs strong selection |
| **Tree-based models** | Robust to redundant features but can overfit with noise |
| **Linear models** | Sensitive to multicollinearity and irrelevant features |
| **Deep learning** | Less sensitive due to automatic feature learning, but high dimensions still affect training time |

</details>
