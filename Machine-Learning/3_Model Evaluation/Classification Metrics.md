# Classification Metrics Cheatsheet (Binary Classification)

This README reorganizes and refines the core ideas from the provided notes on **confusion-matrix-based metrics** and **threshold curves** (ROC / AUC). It also adds practical guidance for **imbalanced classification** and **threshold selection**.

---

<details>
<summary>1) Confusion Matrix (The Foundation)</summary>

A binary classifier produces predictions that fall into four buckets:

|                     | Predicted Negative  | Predicted Positive  |
|---------------------|--------------------:|--------------------:|
| **Actual Negative** | True Negative (TN)  | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP)  |

- **False Positive (FP)** is also called a **Type I error**.
- **False Negative (FN)** is also called a **Type II error**.

</details>

---

<details>
<summary>2) Core Rates & What They Mean</summary>

**Sensitivity / Recall / True Positive Rate (TPR)**
How many actual positives you successfully catch:

$$\mathrm{TPR} = \frac{TP}{TP + FN}$$

**Specificity / True Negative Rate (TNR)**
How many actual negatives you correctly reject:

$$\mathrm{TNR} = \frac{TN}{TN + FP}$$

Synonyms: *specificity, selectivity*

**False Positive Rate (FPR)**
How often you incorrectly flag negatives as positives:

$$\mathrm{FPR} = \frac{FP}{TN + FP} = 1 - \mathrm{Specificity}$$

Synonyms: *Type I error*

**Precision / Positive Predictive Value (PPV)**
Among predicted positives, how many are truly positive:

$$\mathrm{Precision} = \frac{TP}{TP + FP}$$

Synonyms: *Positive Predictive Value*

</details>

---

<details>
<summary>3) Accuracy and Why It Can Mislead</summary>

$$\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Accuracy can look great on **imbalanced data** (e.g., 99% negatives) even if the model misses most positives. Use accuracy when:
- Classes are balanced, **and**
- FP and FN have similar business costs.

*(Scikit-learn scoring name: `accuracy`)*

</details>

---

<details>
<summary>4) F1 Score: Balancing Precision and Recall</summary>

**F1** is the harmonic mean of precision and recall:

$$\mathrm{F1} = 2\cdot\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$$

Best value = 1, worst = 0; precision and recall contribute equally.

Use F1 when:
- You care about **both** missing positives (FN) and creating false alarms (FP)
- The positive class is relatively rare
- You want a single-number summary at a chosen threshold

> If FP is far more expensive than FN (or vice versa), consider **Fβ** — e.g., F0.5 emphasizes precision, F2 emphasizes recall.

*(Scikit-learn scoring name: `f1`)*

</details>

---

<details>
<summary>5) ROC Curve and AUC</summary>

**ROC Curve** is built by sweeping the classification threshold and plotting:
- x-axis: **FPR**
- y-axis: **TPR**

Key landmarks:
- (0, 0): predict everything negative
- (1, 1): predict everything positive
- Diagonal line: **random guess baseline**

**AUC (Area Under ROC)**
- Larger AUC ⇒ better ranking/separation
- Random classifier has AUC ≈ 0.5

**Practical note:** ROC/AUC is often stable to class imbalance since it evaluates *ranking* rather than absolute counts. But it can **overstate usefulness** when the positive class is extremely rare — a small FPR can still create many false positives in absolute terms.

</details>

---

<details>
<summary>6) Precision–Recall Curve (Recommended for Imbalanced Data)</summary>

For heavily imbalanced problems (fraud, churn, anomaly detection), the **PR curve** is often more informative than ROC because it focuses entirely on the positive class:
- Precision falls as you accept more positives (lowering threshold)
- Recall rises as you catch more positives

**Rule of thumb:**
- Use **ROC-AUC** when you care about overall ranking and the dataset isn't extremely imbalanced
- Use **PR-AUC**, **Precision@K**, or **Recall@FPR** when positives are rare and false positives are expensive

</details>

---

<details>
<summary>7) Choosing Metrics by Business Cost</summary>

Ask first: **Which error is worse?**

| Error more costly | Optimize for | Example metric |
|---|---|---|
| **FN** (missing positives) | Recall | F2, high-recall threshold |
| **FP** (false alarms) | Precision | F0.5, high-precision threshold |
| Both matter equally | Balance | F1, PR/ROC operating point |

</details>

---

<details>
<summary>8) Threshold Selection</summary>

Most models output a probability/score. A metric like ROC-AUC is threshold-free, but production needs a threshold.

Common approaches:
1. **Maximize F1** on validation data
2. Fix an acceptable **FPR** (e.g., 1%) and maximize **TPR** at that FPR
3. Fix an operational budget (**top-K** alerts) and maximize **Precision@K**
4. Use **cost-based** thresholding if you can quantify FP/FN costs

> Always choose thresholds on a validation set (or via cross-validation), not on the test set.

</details>

---

<details>
<summary>9) Quick Scikit-learn Examples</summary>

```python
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

# y_true: {0,1}, y_pred: {0,1}, y_score: probability for class 1

cm   = confusion_matrix(y_true, y_pred)
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)

auc             = roc_auc_score(y_true, y_score)
fpr, tpr, _     = roc_curve(y_true, y_score)

precision, recall, _ = precision_recall_curve(y_true, y_score)
ap              = average_precision_score(y_true, y_score)  # PR-AUC
```

</details>
