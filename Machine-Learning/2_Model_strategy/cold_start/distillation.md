# Knowledge Distillation for Cold Start

## The Problem

At cold start, a new entity only has a *subset* of the features that the full model was trained on. Rebuilding a model from scratch on those reduced features works but wastes all the signal the full model has already learned.

**Knowledge distillation** transfers that signal: a *teacher* trained on full features generates soft targets that the *student* (cold-start model) learns to match — giving it access to knowledge it could never derive from the reduced feature set alone.

References:
- https://arxiv.org/abs/1503.02531
- https://www.amazon.science/publications/toward-understanding-privileged-features-distillation-in-learning-to-rank

---

## Regression Distillation (continuous target)

### Step 1 — Train the Teacher on Full Features

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# df has columns: x1..x5, y
X_full = df[["x1","x2","x3","x4","x5"]]
X_cold = df[["x1","x2","x3"]]          # only cold-start features
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

yT_tr = teacher.predict(Xf_tr)   # soft targets on training rows
yT_va = teacher.predict(Xf_va)
print("Teacher RMSE:", mean_squared_error(y_va, teacher.predict(Xf_va), squared=False))
```

### Step 2 — Train the Student on Cold Features with Blended Loss

The student minimizes a blended target that combines the true label and the teacher's soft prediction:

<img width="499" height="39" alt="image" src="https://github.com/user-attachments/assets/ab0de7d0-a96b-4792-aec0-eb8033764320" />

- **α → 1**: student learns from true labels only (no distillation)
- **α → 0**: student learns entirely from the teacher (full distillation)
- **α ≈ 0.5**: balanced; typical starting point

```python
alpha = 0.5   # tune in range 0.2–0.8
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

---

## Key Design Choices

| Choice | Guidance |
|--------|----------|
| **α (blend weight)** | Start at 0.5; increase toward 1 if teacher quality is low |
| **Student depth** | Keep shallower than teacher — it has fewer features, less capacity needed |
| **When to use** | Best when x4, x5 carry real signal that correlates with x1–x3 |
| **When NOT to use** | If cold features are entirely uncorrelated with warm-only features, distillation adds little |
