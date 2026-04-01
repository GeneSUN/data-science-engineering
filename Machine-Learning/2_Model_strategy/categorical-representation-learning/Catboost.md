Here are the most likely CatBoost interview questions:

## Ordered Target Statistics
```
2. Why is CatBoost especially strong on tabular data with many categorical features?
3. How does CatBoost handle categorical variables differently from one-hot encoding?
6. What is “ordered boosting,” and why does it reduce target leakage / prediction shift?
23. How does CatBoost treat high-cardinality categorical features?
```
### 1. The core method: Ordered Target Statistics
- The **Problem** of Target Encoding is target leakage, the model "sees" the answer within the feature itself.
- **CatBoost Fix**, as **ordered boosting** start, it learns the category-target only with rows that appear before in that shuffle.

### 2. Automatic Feature Combination

### 3. Quantization of Categorical Features

### 4. The "One-Hot" Threshold

---

## Oblivious Symmetrics Trees
```
8. What are “oblivious” or “symmetric” trees in CatBoost?
9. What are the advantages and disadvantages of symmetric trees?
12. What is the difference between CatBoost, XGBoost, and LightGBM?
13. When would CatBoost outperform XGBoost or LightGBM?
```
### 1. Fast

### 2. Regularization

---

## Hyperparameters

```
16. What are the most important CatBoost hyperparameters?
17. What do depth, learning_rate, iterations, and l2_leaf_reg do?
18. What do random_strength and bagging_temperature do?
19. What is one_hot_max_size?
20. How do you tune CatBoost for overfitting?
21. What is the overfitting detector / early stopping in CatBoost?
22. How do you use validation sets correctly with CatBoost?
```


1. The "Big Three" (Start Here)
   These have the most direct impact on your model's capacity and error rate.
   - **iterations** (Default: 1000): The maximum number of trees. Unlike XGBoost, CatBoost is quite resistant to high iteration counts if used with early stopping.
   - **learning_rate** (Default: Automatically defined): Small values ($0.01$ to $0.05$) are generally better for accuracy but require more iterations. If you set this manually, you almost always need to tune it alongside iterations.depth (Default: 6):
   - **The depth of the symmetric trees**. Range is 1–16. Since CatBoost uses symmetric trees, a depth of 6-10 is usually the "sweet spot." Anything higher than 10 is very slow and rarely helps unless your data is extremely complex.

2. Categorical Magic
   - "one_hot_max_size: CatBoost uses one-hot encoding for features with a small number of unique values and Target Statistics for the rest.Tip: If you have a category with 5 values and you want it one-hot encoded, set this to 5 or higher.
   - cat_features: This isn't a "tunable" parameter in the traditional sense, but it is the most important: you must pass the indices of your categorical columns here. Don't encode them yourself!

3. Regularization (Fighting Overfitting):
   - l2_leaf_reg: $L_2$ regularization of the leaf values. Higher values (e.g., $3, 5, 10$) make the model more conservative.
   - random_strength: Adds randomness to the scoring of splits. This is unique to CatBoost and helps prevent the model from getting "stuck" on a single feature that looks too good to be true in the training set.
   - bagging_temperature: Controls the intensity of Bayesian bagging.$0$: No bagging (every sample has weight 1).$1$: Standard bagging.Higher values ($>1$): More aggressive sampling, which increases the diversity of the trees.

4. Performance & Hardware
   - task_type: Set to 'GPU' if you have one. CatBoost is arguably the fastest boosting library on GPU.
   - border_count (Default: 254 on CPU, 128 on GPU): The number of splits for numerical features. Increasing this to 254 on GPU can improve accuracy but slows down training.
   - boosting_type:'Ordered': Better for small datasets (uses the "Ordered Boosting" we discussed).'Plain': Faster for large datasets.

