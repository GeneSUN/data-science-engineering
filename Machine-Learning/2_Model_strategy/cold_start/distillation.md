
## distillation

- https://arxiv.org/abs/1503.02531
- https://www.amazon.science/publications/toward-understanding-privileged-features-distillation-in-learning-to-rank

A) Regression distillation (y is continuous)
Step 1) Train the teacher on full features

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# df has columns: x1..x5, y
X_full = df[["x1","x2","x3","x4","x5"]]
X_cold = df[["x1","x2","x3"]]
y = df["y"].values

Xf_tr, Xf_va, Xc_tr, Xc_va, y_tr, y_va = train_test_split(
    X_full, X_cold, y, test_size=0.2, random_state=42
)

teacher = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42
)
teacher.fit(Xf_tr, y_tr)

yT_tr = teacher.predict(Xf_tr)  # teacher "soft targets" on training warm rows
yT_va = teacher.predict(Xf_va)
print("Teacher RMSE:", mean_squared_error(y_va, teacher.predict(Xf_va), squared=False))

```
Step 2) Train the student on x1–x3 to match both y and teacher

Use a blended loss:

<img width="499" height="39" alt="image" src="https://github.com/user-attachments/assets/ab0de7d0-a96b-4792-aec0-eb8033764320" />

```python
alpha = 0.5  # tune this (0.2~0.8 common)
y_blend_tr = alpha * y_tr + (1 - alpha) * yT_tr
y_blend_va = alpha * y_va + (1 - alpha) * yT_va

student = XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42
)
student.fit(Xc_tr, y_blend_tr)

pred = student.predict(Xc_va)
print("Student (distilled) RMSE vs TRUE y:", mean_squared_error(y_va, pred, squared=False))
```


