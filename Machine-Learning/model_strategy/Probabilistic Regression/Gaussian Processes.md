GP is “a probability distribution over functions”


The kernel k(x,x') is the definition of similarity: if two inputs are “similar,” the degree of their outputs correlated.
- close in time → strongly correlated
- far apart → weakly correlated

**Why ```uncertainty``` behaves nicely**
- Near many observations → uncertainty shrinks
- Far from data → uncertainty grows back toward the prior

This “mean + uncertainty band” behavior is a big reason people like GPs.

<img width="592" height="714" alt="Screenshot 2026-03-01 at 4 28 18 PM" src="https://github.com/user-attachments/assets/17c22803-3cd5-4235-b554-85bd27a95f97" />

## Practical
**When GP is a great choice**
- You have small/medium data
- You care about uncertainty
- You want a strong nonlinear model without deep learning
- You want to encode domain beliefs via kernels (smooth/periodic/etc.)

**When GP struggles**
- Big data (classic GP needs an 𝑛×𝑛 kernel matrix and matrix inversion → expensive)
- Very high-dimensional inputs (sometimes works, sometimes not)

<img width="489" height="470" alt="output (2)" src="https://github.com/user-attachments/assets/f378a791-cbbb-4a3f-b4e3-e98d3f839b4a" />
<img width="489" height="470" alt="output (3)" src="https://github.com/user-attachments/assets/a4e63143-8d1b-4c3e-9ddc-c42fc927c776" />

<img width="475" height="690" alt="output (4)" src="https://github.com/user-attachments/assets/289d4603-b468-4177-98d3-0190b6185ce1" />
<img width="474" height="690" alt="output (5)" src="https://github.com/user-attachments/assets/5e820e89-6dde-41cd-9e91-eeae6d0ba86e" />
<img width="456" height="590" alt="output (6)" src="https://github.com/user-attachments/assets/d9295d3c-ada0-454d-9827-a2e52be89869" />


