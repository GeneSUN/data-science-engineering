# Classification Metrics Cheatsheet (Binary Classification)

This README reorganizes and refines the core ideas from the provided notes on **confusion-matrix-based metrics** and **threshold curves** (ROC / AUC). It also adds practical guidance for **imbalanced classification** and **threshold selection**.


---

## 1) Confusion Matrix (The Foundation)

A binary classifier produces predictions that fall into four buckets:

|                | Predicted Negative | Predicted Positive |
|----------------|-------------------:|-------------------:|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

- **False Positive (FP)** is also called a **Type I error**. 
- **False Negative (FN)** is also called a **Type II error**. 

---

## 2) Core Rates & What They Mean

**Sensitivity / Recall / True Positive Rate (TPR)**
How many actual positives you successfully catch:

$$
\mathrm{TPR} = \frac{TP}{TP + FN}
$$


**Specificity / True Negative Rate (TNR)**
How many actual negatives you correctly reject:

$$
\mathrm{TNR} = \frac{TN}{TN + FP}
$$
    
    Synonyms: *specificity, selectivity* 


**False Positive Rate (FPR)**
How often you incorrectly flag negatives as positives:

$$
\mathrm{FPR} = \frac{FP}{TN + FP} = 1 - \mathrm{Specificity}
$$
    
    Synonyms: *Type I error* 



**Precision / Positive Predictive Value (PPV)**
Among predicted positives, how many are truly positive:

$$
\mathrm{Precision} = \frac{TP}{TP + FP}
$$

    Synonyms: *Positive Predictive Value* 

---

## 3) Accuracy and Why It Can Mislead

**Accuracy** is:

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Accuracy can look great on **imbalanced data** (e.g., 99% negatives) even if the model misses most positives. Use accuracy when:
- Classes are balanced, **and**
- FP and FN have similar business costs.

(Scikit-learn scoring name: `accuracy`) 

---

## 4) F1 Score: Balancing Precision and Recall

**F1** is the harmonic mean of precision and recall:

$$
\mathrm{F1} = 2\cdot\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
$$

Interpretation: best value = 1, worst = 0; precision and recall contribute equally. 

Use F1 when:
- You care about **both** missing positives (FN) and creating false alarms (FP),
- The positive class is relatively rare,
- You want a single-number summary at a chosen threshold.

> Note: If FP is far more expensive than FN (or vice versa), consider **Fβ** (e.g., F0.5 emphasizes precision, F2 emphasizes recall).

(Scikit-learn scoring name: `f1`) 

---

## 5) ROC Curve and AUC

### ROC Curve
ROC is built by sweeping the **classification threshold** and plotting:

- x-axis: **FPR**  
- y-axis: **TPR**  

This describes the trade-off between catching positives and raising false alarms as the threshold moves. 

Key landmarks:
- (0, 0): predict everything negative  
- (1, 1): predict everything positive
- diagonal line: **random guess baseline**

### AUC (Area Under ROC)
- Larger AUC ⇒ better ranking/separation 
- Random classifier has AUC ≈ 0.5 

**Practical note (important):** ROC/AUC is often **stable to class imbalance** in the sense that it evaluates *ranking* rather than absolute counts. But it can **overstate usefulness** when the positive class is extremely rare, because a small FPR can still create many false positives in absolute terms.

---

## 6) Precision–Recall Curve (Recommended for Imbalanced Data)

For heavily imbalanced problems (fraud, churn, anomaly detection), the **Precision–Recall (PR) curve** is often more informative than ROC because it focuses on the positive class:
- Precision falls as you accept more positives (lowering threshold),
- Recall rises as you catch more positives.

(Your notes include a PR curve example plot.) 

**Rule of thumb:**
- Use **ROC-AUC** when you care about ranking overall and the dataset isn’t extremely imbalanced.
- Use **PR-AUC**, **Precision@K**, or **Recall@FPR** when positives are rare and false positives are expensive.

---

## 7) Choosing Metrics by Business Cost

Ask first: **Which error is worse?**

- If **FN** is worse (missing positives): optimize **Recall**, or use **F2**, or pick a threshold targeting high recall.
- If **FP** is worse (false alarms): optimize **Precision**, or use **F0.5**, or pick a threshold targeting high precision.
- If both matter similarly: use **F1**, or set an operating point via PR/ROC curves.

---

## 8) Threshold Selection (How to Turn Scores Into Decisions)

Most models output a probability/score. A metric like ROC-AUC is threshold-free, but production needs a threshold.

Common approaches:
1. **Maximize F1** on validation data.
2. Fix an acceptable **FPR** (e.g., 1%) and maximize **TPR** at that FPR.
3. Fix an operational budget (**top-K** alerts) and maximize **Precision@K**.
4. Use **cost-based** thresholding if you can quantify FP/FN costs.

> Always choose thresholds on a validation set (or via cross-validation), not on the test set.

---

## 9) Quick scikit-learn Examples

```python
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

# y_true: {0,1}, y_pred: {0,1}, y_score: probability for class 1

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

auc = roc_auc_score(y_true, y_score)
fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)  # PR-AUC (average precision)
```


