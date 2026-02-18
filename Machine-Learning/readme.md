
## 0) Frame the problem (Business Understanding)

### 0.1. Determine business objectives:

### 0.2. Define the prediction target

### 0.3. Define success metrics

- Common: MAE (easy to interpret in dollars), RMSE (penalizes large errors).
- Often for house prices: train/evaluate on log(price) and use RMSLE / RMSE(log) to reduce the impact of extreme prices (common in practice/competitions).

### *Constraints
- Latency (batch vs real-time), explainability needs, fairness/legal constraints, update frequency.

## 1) Get the raw data (Data Understanding)

## 5) Model


### 5.1. Global Model


### 5.2. Local Model

1. How to define similiarity
2. Curse of high dimension

### 5.3. Cluster/Local Regression

---

## 2) Split strategy (before heavy modeling)

### I). Train/validation/test split

- Random split if data is i.i.d.
- Time-based split if market shifts (train on past, test on future).

### II). Group leakage
- If multiple sales per property, split by property_id so the same house doesn’t appear in train and test.

## 3) Clean + preprocess (Data Preparation)

Key idea: build preprocessing so it’s identical at training and serving (avoid training-serving skew).

1. Fix types & units

- Numeric stored as strings, mixed units, etc.

2. Handle missing values

- Numeric: median/constant + missing indicator
- Categorical: “Unknown”

3. Encode categoricals

- One-hot encoding is a strong default for tabular housing data.

4. Scale numeric features

- Often helpful for linear models; not required for tree/boosting models.

## 4) Feature engineering that usually matters







