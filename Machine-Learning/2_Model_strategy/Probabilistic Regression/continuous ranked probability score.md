# Understanding CRPS (Continuous Ranked Probability Score)  
### for Probabilistic Regression

---

## 1️⃣ Predicted Distribution: Mean and Uncertainty

Firstly, let’s go back to the predicted distribution.  
For this distribution, there are two critical parameters, **μ** and **σ**.

### 🔹 μ — Conditional Mean
- μ represents the conditional mean.
- You want μ to be as close to `y_actual` as possible, because that controls where the bulk of the probability mass is centered.

### 🔹 σ — Conditional Uncertainty (Spread)
- σ represents the conditional uncertainty (spread).
- You generally want σ to be as small as possible while still being realistic, because a smaller σ means a sharper (more confident) distribution.

---

## 2️⃣ How μ and σ Determine Exceedance Probability

μ and σ together determine the exceedance probability.

For example, if `y_actual = 10` and the model predicts `μ = 9`:

- If σ is small  
  - Most probability mass stays near 9–10  
  - `P(Y > 30)` will be very small  
  - High confidence it won’t exceed 30  

- If σ is large  
  - The distribution has heavier tail mass beyond 30  
  - `P(Y > 30)` increases  
  - Less confidence  

### ⚖️ Trade-off

Making σ too small is risky:

- If μ is even slightly off from `y_actual`, the model becomes overconfident
- It gets heavily penalized because it assigned very low probability to what actually happened

So the model must balance:

> **Sharpness (small σ)**  
> vs  
> **Accuracy & calibration (not being overconfident when uncertain)**

---

## 3️⃣ Validation: Evaluate the Distribution

When validating, you should evaluate the **distribution**, not just a point.

Key questions:

- Does it put high mass near the realized `y`? *(location / fit)*  
- Is its uncertainty appropriate? *(calibration / sharpness)*  

---

## 4️⃣ CRPS: Comparing Predicted vs Empirical CDF

$$
mathrm{CRPS}(F, y) =
\int_{-\infty}^{\infty}
\big(F(t) - F_{\mathrm{emp}}(t)\big)^2 \, dt
$$

Where:

- $F(t)$ = predicted CDF  
- $F_{\mathrm{emp}}(t)$ = empirical CDF (step at $y$)



### 📌 Empirical CDF
The empirical CDF can be viewed as a point-mass at the observed value.

- It puts all mass at `y_actual`
- It is a “delta” distribution
- It corresponds to a step CDF
- Conceptually has zero spread

### 📌 Predicted CDF
The Continuous Ranked Probability Score compares how different:

> **Predicted CDF**  
> vs  
> **Empirical CDF** (step function jumping at `y_actual`)

If:

- μ is close to `y_actual`
- σ is appropriately sharp

Then:

> The predicted CDF transitions near that jump  
> → CRPS becomes small

---

### 📊 Visual Comparison
CRPS evaluates both location and spread together.  

| Correct center, small variance | Correct center, large variance | Wrong center, small variance |
|--------|--------|--------|
| <img src="https://github.com/user-attachments/assets/cb76e4e3-d733-4d57-880e-403c2e8ac195" width="100%"> | <img src="https://github.com/user-attachments/assets/032ef9a8-cf9f-4bea-9049-c6103966e316" width="100%"> | <img src="https://github.com/user-attachments/assets/cb57d69e-73ce-4467-8172-f0e5e9b8da45" width="100%"> |

---

## 5️⃣ What Minimizing CRPS Encourages

Minimizing CRPS encourages two things simultaneously:

- The center of the predicted distribution should align with `y_actual`  
  *(good μ-like behavior)*

- The distribution should be sharp but not miscalibrated  
  *(good σ-like behavior)*

---




