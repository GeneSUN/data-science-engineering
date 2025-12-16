# Feature engineering

## 1. Start with Domain Knowledge
Before applying any statistical or machine learning technique, start by consulting:
	• Business context and subject matter experts
	• Understand which features are likely to be relevant or redundant
	• Identify data quality issues, potential leakage features, and proxy variables

```python
feature_groups = {
                    
                    "signal_quality": [
                        "4GRSRP", "4GRSRQ", "SNR", "4GSignal", "5GRSRP", "5GRSRQ", "5GSNR","BRSRP", "RSRQ", "CQI",
                    ],
                    
                    "hardware_health": [
                        "4GAntennaTemp", "4GAntennaTempThreshold",
                        "5GNRSub6AntennaTemp", "5GNRSub6AntennaTempThreshold",
                        ...
                    ],
                    
                    "throughput_data": [
                        "LTEPDSCHPeakThroughput", "LTEPDSCHThroughput",
                        "LTEPUSCHPeakThroughput", "LTEPUSCHThroughput",
                        "TxPDCPBytes", "RxPDCPBytes",
                        ...
                    ],
                    
                    "gps_location": [
                        "GPSLatitude", "GPSLongitude", "GPSAltitude", "GPSEnabled",
                        "HomeRoam"
                    ],
                    
                    "device_static_info": [
                        "IMEI", "IMSI", "MDN", "sn", "mac", "rowkey",
                        ...
                    ],
                    
                    "uptime_downtime_status": [
                        "uptime", "ServiceUptime", "ServiceUptimeTimestamp",
                        "ServiceDowntime", "ServiceDowntimeTimestamp",
                        "5GUptimeTimestamp", "5GDowntimeTimestamp", "Status"
                    ],
                    
                    "mobility_stability": [
                        "LTEHandOverAttemptCount", "LTEHandOverFailureCount",
                        "LTERACHAttemptCount", "LTERACHFailureCount",
                        ...
                    ],
                    
}

```

- https://colab.research.google.com/drive/1s7sReQ_0cpIsWHwGSSls5PVenGyayZ1T#scrollTo=BLkQ6zYT0NvN

```python

from dataclasses import MISSING, dataclass, field
from typing import List, Optional, Iterable
import pandas as pd
import copy

@dataclass
class FeatureConfig:

    date: List = field(
        default=MISSING,
        metadata={"help": "Column name of the date column"},
    )
    target: str = field(
        default=MISSING,
        metadata={"help": "Column name of the target column"},
    )

    original_target: str = field(
        default=None,
        metadata={
            "help": "Column name of the original target column in acse of transformed target. If None, it will be assigned same value as target"
        },
    )

    continuous_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the categorical fields. Defaults to []"},
    )
    boolean_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the boolean fields. Defaults to []"},
    )

    index_cols: str = field(
        default_factory=list,
        metadata={
            "help": "Column names which needs to be set as index in the X and Y dataframes."
        },
    )
    exogenous_features: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names of the exogenous features. Must be a subset of categorical and continuous features"
        },
    )
    feature_list: List[str] = field(init=False)

    def __post_init__(self):
        assert (
            len(self.categorical_features) + len(self.continuous_features) > 0
        ), "There should be at-least one feature defined in categorical or continuous columns"
        self.feature_list = (
            self.categorical_features + self.continuous_features + self.boolean_features
        )
        ...

    def get_X_y(
        self, df: pd.DataFrame, categorical: bool = False, exogenous: bool = False
    ):
        ...
        return X, y, y_orig

# ------------------------------------------------------------------------------
feat_config = FeatureConfig(
    date="timestamp",
    target="energy_consumption",
    continuous_features=[
        "visibility",
        "windBearing",
        "temperature",
        "dewPoint",
        "pressure",
        "apparentTemperature",
        "windSpeed",
        "humidity",
        "energy_consumption_lag_1",
        "energy_consumption_lag_2",
        "energy_consumption_lag_3",
       ...
    ],
    categorical_features=[
        "holidays",
        "precipType",
        "icon",
        "summary",
        "timestamp_Month",
        "timestamp_Quarter",
        "timestamp_WeekDay",
        "timestamp_Dayofweek",
        "timestamp_Dayofyear",
        "timestamp_Hour",
        "timestamp_Minute",
    ],
    boolean_features=[
        "timestamp_Is_quarter_end",
        "timestamp_Is_quarter_start",
        "timestamp_Is_year_end",
        "timestamp_Is_year_start",
        "timestamp_Is_month_start",
    ],
    index_cols=["timestamp"],
    exogenous_features=[
        "holidays",
        "precipType",
        "icon",
        "summary",
        "visibility",
        "windBearing",
        "temperature",
        "dewPoint",
        "pressure",
        "apparentTemperature",
        "windSpeed",
        "humidity",
    ],
)

```

---

## 2. Statistical Analysis
	a. EDA-Distribution of features, 
		○ features with low variance
		○ Feature with too many missing values
	b. Feature-Target Correlation
		○ Use correlation coefficients (Pearson, Spearman) for numerical targets
		○ Use Chi-squared test or Mutual Information for categorical targets
	c. Multicollinearity Check
		○ Compute correlation matrix or Variance Inflation Factor (VIF) to detect highly correlated features
		○ Drop or combine redundant variables

---

## 3. Machine Learning-Based Feature Engineering

	a. Subset Selection
		○ Use Forward Selection, Backward Elimination, or Recursive Feature Elimination (RFE)
    
	b. Regularization Methods
		○ Use Lasso (L1) or ElasticNet to automatically shrink unimportant features
		○ These methods are especially useful in high-dimensional data

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.1,
    random_state=42
)
model.fit(X, y)


coef = pd.Series(model.coef_[0], index=feature_names)

selected_features = coef[coef != 0].index.tolist()
removed_features = coef[coef == 0].index.tolist()
```
  
	c. Dimensionality Reduction
		○ Use PCA or Truncated SVD for unsupervised feature compression
		○ Useful when dealing with highly correlated numeric variables

## 4. Post-Modeling Feature Selection

After model training, you can analyze:

	a. Model-Based Feature Importance
		○ Tree-based models (Random Forest, XGBoost, etc.) naturally provide importance scores
		○ Use SHAP values or permutation importance for more interpretability
	b. Statistical Significance (for linear models)
		○ Look at p-values from Generalized Linear Models (GLM) or logistic regression to assess significance
		○ Beware of multicollinearity or sample size affecting p-values

## 5. Collaboration and Feedback
	• Present selected features and their importance to:
		○ Business stakeholders (validate if the features make sense)
		○ Engineering or domain experts (check for technical relevance)
	• Use this feedback loop to adjust, refine, or re-engineer features

## 6. Consider Model-Specific Needs
Different models have different sensitivities:

| Model Family                                                   | Feature Sensitivity                                                                          | Why This Happens                                                                                                                                     | 
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | 
| **Tree-Based Models (e.g., Random Forest)**                    | **Moderately robust**                           | Trees split greedily on impurity reduction;                           |
| **Gradient Boosted Trees (e.g., XGBoost, LightGBM)**           | **More sensitive than Random Forests** | Boosting sequentially fits residuals, so noise can be repeatedly amplified across trees                                                              |
| **General Linear Models (e.g., Linear / Logistic Regression)** | **Highly sensitive**                          | Coefficients are estimated globally; correlated features lead to unstable or inflated weights                                                        | 
| **Neural Networks**                                            | **Less sensitive**, but high-dimensional noise increases cost | Hidden layers learn nonlinear feature combinations, but optimization cost and data requirements grow with feature count                              | 

Some models benefit from more engineered features, while others perform better with raw data and embedded learning.

## Summary
**Post-Modeling Feature Selection** and **Subset Selection** are the most common methods i used.
