# Feature engineering



---

## 0. Cheap “pre-selection”
- Remove near-constant / low-variance features
  - categorical feature: entropy/gini index
  - variance
  	- ```sklearn.feature_selection.VarianceThreshold(threshold=0.0) ```
- Handle multicollinearity / redundancy
	- Compute correlation matrix 
	- Variance Inflation Factor (VIF)
- Missingness-driven pruning


### how to handle multicollinearity?
PCA is good example, when 
- features are highly correlated / redundant
- relationship is mostly linear
- mainly care about prediction, not interpretation
- train/test distribution shifts small
  
<img width="735" height="326" alt="image" src="https://github.com/user-attachments/assets/a08aa6cf-fb61-43ca-99e0-1fc430bd777e" />


---

## 2. Wrapper methods

firstly check Model-Based Feature Importance
	- Tree-based models (Random Forest, XGBoost, etc.) naturally provide importance scores
	- Use SHAP values or permutation importance for more interpretability

for careful selection, Wrapper methods: 
	- Use Forward/Backward Elimination,  
	- Recursive Feature Elimination (RFE)
		- ```sklearn.feature_selection.RFE```
		- ```sklearn.feature_selection.RFECV```

## 3. Embedded methods (selection happens inside training)

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

## 4. Stability strategies

Idea: features that look good once might be unstable. especially for time series analysis, feature importance change :
- Run selection over bootstraps / CV folds / time splits
- Keep features that are consistently selected (stability selection / consensus selection)

---






## 5. Consider Model-Specific Needs
Different models have different sensitivities:

| Model Family                                                   | Feature Sensitivity                                                                          | Why This Happens                                                                                                                                     | 
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | 
| **Tree-Based Models (e.g., Random Forest)**                    | **Moderately robust**                           | Trees split greedily on impurity reduction;                           |
| **Gradient Boosted Trees (e.g., XGBoost, LightGBM)**           | **More sensitive than Random Forests** | Boosting sequentially fits residuals, so noise can be repeatedly amplified across trees                                                              |
| **General Linear Models (e.g., Linear / Logistic Regression)** | **Highly sensitive**                          | Coefficients are estimated globally; correlated features lead to unstable or inflated weights                                                        | 
| **Neural Networks**                                            | **Less sensitive**, but high-dimensional noise increases cost | Hidden layers learn nonlinear feature combinations, but optimization cost and data requirements grow with feature count                              | 

Some models benefit from more engineered features, while others perform better with raw data and embedded learning.

## Categorical feature selections

## Summary
**Post-Modeling Feature Selection** and **Subset Selection** are the most common methods i used.


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
