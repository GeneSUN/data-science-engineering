
## Categorical Features
- https://github.com/GeneSUN/data-science-engineering/tree/main/categorical-representation-learning

## Ordinal Encoding Methods

### sklearn.OrdinalEncoder

### Target-Based Ordinal Encoding

- Rank categories by this statistic
- Particularly powerful when the "true" order is unclear
- category_encoders.TargetEncoder with proper cross-validation

More advanced version, ```category_encoders.glmm.GLMMEncoder```

### Monotonic Constraints (Tree Models)

- Instead of encoding, tell the model the feature is ordinal
- LightGBM: monotone_constraints parameter
- XGBoost: monotone_constraints parameter




