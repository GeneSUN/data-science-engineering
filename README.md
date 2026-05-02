# Data Science Engineering

A practitioner's reference on machine learning and deep learning — organized around how models are actually built, trained, and evaluated.

---

## Machine Learning

→ [Machine Learning](Machine-Learning/readme.md)

- **Preprocessing**
    - Imputation — strategies for missing data
    - Categorical encoding — one-hot, ordinal, target encoding
    - Imbalanced datasets — SMOTE, over/under-sampling
    - Scaling — normalization, standardization
- **Modeling Strategies**
    - Categorical representation learning
        - CatBoost — native categorical handling
        - TabTransformer — transformer-based feature learning
        - Self-supervised pretraining — BERT-style categorical embeddings
    - Cold start — predictions for new entities without historical data
        - Similarity priors, hierarchical Bayes, distillation
        - Prediction with missing covariates
    - Probabilistic regression — uncertainty quantification
        - Conformal prediction, quantile regression, parametric, Gaussian Processes
- **Evaluation**
    - Feature importance — SHAP, permutation, tree-based
    - Feature selection
    - Classification metrics — precision, recall, F1, ROC-AUC


---

## Deep Learning

→ [Deep Learning](Deep-Learning/readme.md)

- **MLP** — fully connected networks, activation functions
- **Neural Network Architecture** — architectural families and design patterns
    - Attention — self-attention, multi-head attention mechanisms
    - RNN-based — LSTM, GRU, sequence modeling
    - CNN, Transformer overviews
- **Optimization** — why training fails and how to fix it
    - Core problems
        - Vanishing / exploding gradients
        - Loss landscape — non-convex geometry, saddle points, plateaus
        - Overfitting — generalization gaps
        - Hyperparameter sensitivity — fragility to initialization, learning rate, batch size
    - Techniques
        - Initialization — Xavier, He
        - Normalization — BatchNorm, LayerNorm
        - Optimizers — SGD, Momentum, Adam, adaptive algorithms
        - Regularization — Dropout, weight decay, early stopping
        - Learning rate scheduling — warmup, cosine decay, step schedules

---

## Notebooks

| Notebook | Topic |
|---|---|
| [categorical_modeling.ipynb](Notebook/categorical_modeling.ipynb) | OHE, target encoding, embeddings, TabTransformer |
| [Hierarchical_Models_with_Predictors.ipynb](Notebook/Hierarchical_Models_with_Predictors.ipynb) | Bayesian hierarchical modeling for sparse data |
| [trip_duration.ipynb](Notebook/trip_duration.ipynb) | Regression example: trip duration prediction |
