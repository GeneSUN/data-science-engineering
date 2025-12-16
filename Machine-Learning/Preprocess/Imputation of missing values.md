# Imputation of missing values

## Reason for missing values

Typical imputation methods tend to assume that the data is Missing Completely at Random (MCAR) or Missing at Random (MAR), but when the data is Missing Not at Random (MNAR), then the imputation methods kill the information contained in the original feature. 

##  Imputation Techniques
- Univariate imputes values in the i-th feature dimension using only non-missing values in that feature dimension (e.g. impute.SimpleImputer). 
- Multivariate imputation algorithms use the entire set of available feature dimensions to estimate the missing values (e.g. impute.IterativeImputer).

**Univariate feature imputation**

- The ```python sklearn.impute.SimpleImputer``` provides basic strategies for imputing missing values. 
- Missing values can be imputed with a provided constant value, or statistics (mean, median or most frequent) of each column in which the missing values are located. 

**Multivariate feature imputation**
- A more sophisticated approach is to use the IterativeImputer class, which models each feature with missing values as a function of other features, and uses that estimate for imputation. 
- The IterativeImputer class is very flexible - it can be used with a variety of estimators to do round-robin regression, treating every variable as an output in turn.

**Nearest neighbors imputation**

Each sample’s missing values are imputed using the mean value from n_neighbors nearest neighbors found in the training set. Two samples are close if the features that neither is missing are close.
The feature of the neighbors are averaged uniformly or weighted by distance to each neighbor.

Other estimators for impute missing value
- [How to Use Python and MissForest Algorithm to Impute Missing Data](https://towardsdatascience.com/how-to-use-python-and-missforest-algorithm-to-impute-missing-data-ed45eb47cb9a/)
- [The MICE Algorithm](https://cran.r-project.org/web/packages/miceRanger/vignettes/miceAlgorithm.html)

**XGBoost**

XGBoost can natively handle missing values during training. Its underlying decision trees are effectively trinary at each split, routing observations with missing feature values along a learned default direction. In this way, missingness is treated as an informative signal rather than noise.

## Dummy imputation-Marking imputed values

This approach allows downstream models to learn missingness patterns explicitly, which is particularly important when the data is **Missing Not At Random (MNAR)**.

<img width="300" height="386" alt="image" src="https://github.com/user-attachments/assets/d6d526bb-e192-4b9a-accf-40dc3e3a4486" />


The ```MissingIndicator``` transformer is useful to transform a dataset into corresponding binary matrix indicating the presence of missing values in the dataset.
