## 1) Frame the Problem (Business Understanding)

<details>

### 1.1 Determine Business Objectives
- What is the goal? (e.g., accurate pricing, ranking, decision support)
- Who uses the prediction? (buyers, sellers, internal systems)
- What action will be taken based on predictions?

### 1.2 Define the Prediction Target
- Predict raw price vs log(price)
- Regression vs ranking vs classification (e.g., price range)

### 1.3 Define Success Metrics
- MAE → easy to interpret (dollar error)
- RMSE → penalizes large errors
- RMSLE / RMSE(log(price)) → reduces impact of extreme prices (common in practice)

### 1.4 Constraints
- Latency: batch vs real-time
- Explainability requirements
- Fairness / legal constraints
- Update frequency (daily, weekly, etc.)

</details>

---

## 2) Get the Raw Data (Data Understanding)

<details>

- Identify data sources (transactions, property features, location data, external signals)
- Understand schema and data granularity
- Perform initial EDA:
  - Distribution of price
  - Missingness patterns
  - Outliers
- Check data quality issues early

</details>

---

## 3) Split Strategy (Before Heavy Modeling)

<details>

### 3.1 Train / Validation / Test Split
- Random split if data is i.i.d.
- Time-based split if market shifts (train on past → test on future)

### 3.2 Group Leakage
- If multiple sales per property:
  - Split by `property_id`
  - Avoid same house appearing in both train and test

</details>

---

## 4) Clean + Preprocess (Data Preparation)

<details>

**Key idea:** Ensure preprocessing is identical in training and serving (avoid training-serving skew)

### 4.1 Fix Types & Units
- Convert numeric stored as strings
- Standardize units (e.g., sqft vs m²)

### 4.2 Handle Missing Values
- Numeric: median / constant + missing indicator
- Categorical: "Unknown"

### 4.3 Encode Categoricals
- One-hot encoding is a strong default for tabular data

### 4.4 Scale Numeric Features
- Helpful for linear models
- Not required for tree-based models

</details>

---

## 5) Feature Engineering (High Impact)

<details>

- Location-based features (zipcode, neighborhood stats)
- Price per sqft
- Age of property (current_year - built_year)
- Interaction features (e.g., location × size)
- Aggregations (avg price in area, historical trends)
- Temporal features (market trend, seasonality)

</details>

---

## 6) Modeling

<details>

### 6.1 Global Model
- Single model trained on all data
- Examples:
  - Linear regression
  - Gradient boosting (XGBoost, LightGBM)
  - Neural networks

### 6.2 Local Model
- Train models per segment or use nearest neighbors

Key challenges:
1. How to define similarity
2. Curse of dimensionality

### 6.3 Cluster-Based / Local Regression
- Cluster data first (e.g., by location or price range)
- Train separate models per cluster

</details>

---

## 7) Evaluation

