# CatBoost Interview Questions

---

## 1. Ordered Target Statistics

> *Related questions: Why is CatBoost especially strong on tabular data with many categorical features? How does CatBoost handle categorical variables differently from one-hot encoding? What is "ordered boosting," and why does it reduce target leakage / prediction shift? How does CatBoost treat high-cardinality categorical features?*

### 1.1 The Core Method: Ordered Target Statistics
- **Problem with Target Encoding** — target leakage: the model "sees" the answer within the feature itself.
- **CatBoost Fix** — as ordered boosting starts, it learns the category-target relationship only with rows that appear *before* the current row in that shuffle.

### 1.2 Automatic Feature Combination

### 1.3 Quantization of Categorical Features

### 1.4 The "One-Hot" Threshold

---

## 2. Oblivious Symmetric Trees

> *Related questions: What are "oblivious" or "symmetric" trees in CatBoost? What are the advantages and disadvantages of symmetric trees? What is the difference between CatBoost, XGBoost, and LightGBM? When would CatBoost outperform XGBoost or LightGBM?*

### 2.1 Fast

### 2.2 Regularization

---

## 3. Hyperparameters

> *Related questions: What are the most important CatBoost hyperparameters? What do depth, learning_rate, iterations, and l2_leaf_reg do? What do random_strength and bagging_temperature do? What is one_hot_max_size? How do you tune CatBoost for overfitting? What is the overfitting detector / early stopping in CatBoost? How do you use validation sets correctly with CatBoost?*

### 3.1 The "Big Three" — Start Here
These have the most direct impact on your model's capacity and error rate.

- **iterations** (Default: 1000) — the maximum number of trees. Unlike XGBoost, CatBoost is quite resistant to high iteration counts if used with early stopping.
- **learning_rate** (Default: auto) — small values (0.01–0.05) are generally better for accuracy but require more iterations. If set manually, always tune alongside `iterations`.
- **depth** (Default: 6) — depth of the symmetric trees; range is 1–16. Since CatBoost uses symmetric trees, 6–10 is usually the sweet spot. Anything above 10 is very slow and rarely helps unless data is extremely complex.

### 3.2 Categorical Magic

- **one_hot_max_size** — CatBoost uses one-hot encoding for features with a small number of unique values and Target Statistics for the rest. If you have a category with 5 values and want it one-hot encoded, set this to 5 or higher.
- **cat_features** — not tunable in the traditional sense, but the most important: pass the indices of your categorical columns here. Do not encode them yourself.

### 3.3 Regularization — Fighting Overfitting

- **l2_leaf_reg** — L2 regularization of the leaf values. Higher values (e.g., 3, 5, 10) make the model more conservative.
- **random_strength** — adds randomness to the scoring of splits. Unique to CatBoost; prevents the model from over-relying on a single feature that looks too good in the training set.
- **bagging_temperature** — controls the intensity of Bayesian bagging. `0`: no bagging. `1`: standard bagging. Values `> 1`: more aggressive sampling, increasing tree diversity.

### 3.4 Performance & Hardware

- **task_type** — set to `'GPU'` if available. CatBoost is arguably the fastest boosting library on GPU.
- **border_count** (Default: 254 on CPU, 128 on GPU) — number of splits for numerical features. Increasing to 254 on GPU can improve accuracy but slows training.
- **boosting_type** — `'Ordered'`: better for small datasets (uses Ordered Boosting). `'Plain'`: faster for large datasets.
